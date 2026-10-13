"""A verified kernel checkpoint driven through the ``epor-chat-v1`` format.

Every alignment component that runs a model — teacher sampling, critique and
revision, the AI judge, model-written evaluation items, overoptimization
monitors, and the model-level G10 suite — goes through :class:`ChatModel`:

* the checkpoint is hash-verified and identified by the SHA-256 of its
  ``checkpoint.json``, its config hash, and its build's tokenizer ID, which
  become the :class:`~epor.alignment.records.Generator` of anything it writes;
* generation may emit only learned byte/merge IDs or a stop ID
  (``<|end_turn|>``, ``<|eos|>``), so a model can never write a role token and
  forge a turn;
* sampling is greedy, or tempered top-k drawn from a dedicated seeded
  ``torch.Generator`` and never the global RNG.

Outputs are model text from a local research checkpoint: data to be judged,
audited, or measured, never advice and never an instruction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from epor.models.reference import EporDecoder
from epor.tokenizer import EporTokenizer

from .chat import STOP_IDS, ChatMessage, encode_prompt, encode_response
from .records import Generator, Lineage, Sampling


@dataclass
class ChatModel:
    model: EporDecoder
    tokenizer: EporTokenizer
    generator: Generator
    checkpoint_dir: Path
    trainer: dict[str, Any]

    @property
    def checkpoint_sha256(self) -> str:
        assert self.generator.checkpoint_sha256 is not None
        return self.generator.checkpoint_sha256

    @property
    def max_length(self) -> int:
        return self.model.config.configured_max_context

    def lineage(self) -> Lineage:
        return Lineage(
            kind="checkpoint", id=self.checkpoint_dir.name, sha256=self.checkpoint_sha256
        )

    def _allowed(self) -> torch.Tensor:
        mask = torch.full((self.model.config.vocab_size,), float("-inf"))
        mask[256 : self.tokenizer.learned_vocab_size] = 0.0
        for stop in STOP_IDS:
            mask[stop] = 0.0
        return mask

    @torch.no_grad()
    def generate(
        self, messages: Sequence[ChatMessage], sampling: Sampling
    ) -> tuple[str, tuple[int, ...]]:
        """Sampled assistant reply text and its exact token IDs (without the stop ID)."""

        if sampling.decoding not in {"greedy", "temperature"}:
            raise ValueError("a local model samples greedily or with a temperature")
        prompt = encode_prompt(self.tokenizer, messages)
        limit = sampling.max_new_tokens if sampling.max_new_tokens is not None else 64
        generator = torch.Generator(device="cpu")
        generator.manual_seed(sampling.seed or 0)
        allowed = self._allowed()
        produced: list[int] = []
        self.model.eval()
        for _ in range(limit):
            window = (prompt + produced)[-self.max_length :]
            logits = self.model(torch.tensor([window], dtype=torch.long)).logits[0, -1].float()
            logits = logits + allowed
            if sampling.decoding == "greedy" or not sampling.temperature:
                next_id = int(torch.argmax(logits))
            else:
                logits = logits / sampling.temperature
                if sampling.top_k is not None:
                    threshold = torch.topk(logits, min(sampling.top_k, logits.numel())).values[-1]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                next_id = int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
            if next_id in STOP_IDS:
                break
            produced.append(next_id)
        return self.tokenizer.decode(produced), tuple(produced)

    @torch.no_grad()
    def response_logprobs(
        self, requests: Sequence[tuple[Sequence[ChatMessage], str]]
    ) -> list[tuple[float, int]]:
        """``(sum log p, token count)`` of each reply after its prompt, ``<|end_turn|>`` included.

        Prompts longer than the context are truncated from the left; a reply
        that alone does not fit is an error.
        """

        results: list[tuple[float, int]] = []
        for messages, response in requests:
            prompt = encode_prompt(self.tokenizer, messages)
            reply = encode_response(self.tokenizer, response)
            if len(reply) + 1 > self.max_length:
                raise ValueError("a reply does not fit in the model's context")
            ids = (prompt + reply)[-(self.max_length + 1) :]
            inputs = torch.tensor([ids[:-1]], dtype=torch.long)
            logprobs = F.log_softmax(self.model(inputs).logits[0].float(), dim=-1)
            targets = torch.tensor(ids[-len(reply) :], dtype=torch.long)
            positions = torch.arange(len(ids) - 1 - len(reply), len(ids) - 1)
            total = float(logprobs[positions].gather(1, targets[:, None]).sum())
            results.append((total, len(reply)))
        return results


def checkpoint_identity(directory: Path) -> str:
    from epor.training.run_checkpoint import MANIFEST_FILE

    return hashlib.sha256((directory / MANIFEST_FILE).read_bytes()).hexdigest()


def load_chat_model(checkpoint: str | Path, data_root: str | Path) -> ChatModel:
    """Load a verified kernel (or alignment) checkpoint and its build's tokenizer."""

    from epor.models.config import model_config_from_mapping
    from epor.reference_policy import validate_kernel_model
    from epor.tokenizer import load_tokenizer
    from epor.tokenizer.shards import load_tokenized_manifest
    from epor.training.distributed import DistributedContext
    from epor.training.kernel import PackedTrainingSettings
    from epor.training.run_checkpoint import load_kernel_checkpoint, read_kernel_checkpoint

    trainer = read_kernel_checkpoint(checkpoint)
    directory = Path(trainer["checkpoint_dir"])
    config = validate_kernel_model(model_config_from_mapping(trainer["model_config"]))
    settings = PackedTrainingSettings.model_validate(trainer["settings"])
    root = Path(data_root).expanduser().resolve()
    tokenizer = load_tokenizer(
        root, load_tokenized_manifest(root, settings.tokenized_id).tokenizer_id
    )
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("checkpoint vocabulary does not match its build's tokenizer")
    model = EporDecoder(config, seed=0, device="cpu", dtype=torch.float32, initialize=False)
    load_kernel_checkpoint(
        directory, model=model, optimizer=None, context=DistributedContext(), restore_rng=False
    )
    model.eval().requires_grad_(False)
    identity = checkpoint_identity(directory)
    return ChatModel(
        model=model,
        tokenizer=tokenizer,
        generator=Generator(
            kind="epor-checkpoint",
            provider="EPOR local checkpoint",
            model=config.slug,
            version=f"step-{int(trainer['step'])}",
            checkpoint_sha256=identity,
            config_sha256=config.sha256,
            tokenizer_id=tokenizer.tokenizer_id,
        ),
        checkpoint_dir=directory,
        trainer=trainer,
    )


__all__ = ["ChatModel", "checkpoint_identity", "load_chat_model"]
