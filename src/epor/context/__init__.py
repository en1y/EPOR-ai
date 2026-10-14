"""The v0.0.8 context laboratory.

``extension``   derived configs for progressive extension and the sparse retrofit
``tasks``       locally generated G6 items: passkey, RULER-style retrieval,
                multi-document QA, long code, and prompt-injection resistance
``g6``          the G6 suite runner, its report, and its gate
``probe``       TTFT, decode throughput, peak RSS, and OOM behaviour per length,
                each measured in a bounded subprocess
``sweep``       window, local:global ratio, and split-RoPE-base sweeps
``projection``  32K/128K/256K recipe projections from operator inputs
``status``      configured, trained, and validated context, kept distinct

Nothing here raises a model's validated context except a passing G6 report at
that length, and no measurement on the CPU reference machine is a claim about
any release-scale model.
"""
