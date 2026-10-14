"""Sources cited by the context laboratory.

``gemma-3-technical-report``, ``deepseek-v2``, and ``deepseek-v3-2`` are
catalog keys in ``research/catalog.yaml``; RULER, passkey retrieval, position
interpolation, and YaRN are not in the catalog yet and are cited by title and
arXiv ID.  Each published method is ``disclosed``; its use in EPOR is labelled
where it is applied.
"""

from __future__ import annotations

from epor.scaling.report import Citation

RULER = Citation(
    key="ruler",
    title="RULER: What's the Real Context Size of Your Long-Context Language Models?",
    reference="arXiv:2404.06654",
    label="disclosed",
    use="Task design for multi-key needle retrieval at controlled depths (EPOR generators).",
)
PASSKEY = Citation(
    key="landmark-attention-passkey",
    title="Landmark Attention: Random-Access Infinite Context Length for Transformers",
    reference="arXiv:2305.16300",
    label="disclosed",
    use="The passkey retrieval task design (EPOR generator).",
)
POSITION_INTERPOLATION = Citation(
    key="position-interpolation",
    title="Extending Context Window of Large Language Models via Positional Interpolation",
    reference="arXiv:2306.15595",
    label="disclosed",
    use="Linear RoPE position interpolation.",
)
YARN = Citation(
    key="yarn",
    title="YaRN: Efficient Context Window Extension of Large Language Models",
    reference="arXiv:2309.00071",
    label="disclosed",
    use="Per-dimension RoPE interpolation ramp and attention temperature.",
)
GEMMA3 = Citation(
    key="gemma-3-technical-report",
    title="Gemma 3 Technical Report",
    reference="arXiv:2503.19786",
    label="disclosed",
    use="Interleaved bounded-window local and global attention with a split RoPE base.",
)
DEEPSEEK_V2 = Citation(
    key="deepseek-v2",
    title="DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
    reference="arXiv:2405.04434",
    label="disclosed",
    use="Multi-head latent attention and its compressed KV cache.",
)
DEEPSEEK_V32 = Citation(
    key="deepseek-v3-2",
    title="DeepSeek-V3.2",
    reference="arXiv:2512.02556",
    label="disclosed",
    use="Sparse attention through a lightweight indexer and top-k selection, as a retrofit.",
)

__all__ = [
    "DEEPSEEK_V2",
    "DEEPSEEK_V32",
    "GEMMA3",
    "PASSKEY",
    "POSITION_INTERPOLATION",
    "RULER",
    "YARN",
]
