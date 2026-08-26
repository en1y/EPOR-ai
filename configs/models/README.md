# Model configurations

These five files are executable architecture contracts, not released checkpoints
or benchmark claims:

- `epor-tiny.yaml`: roughly 115K parameters for fast CI only.
- `epor-reference.yaml`: canonical 10.4M-parameter pure-PyTorch reference.
- `epor-alpha.yaml`: planned EPOR-α target configuration.
- `epor-beta.yaml`: planned EPOR-β target configuration.
- `epor-gamma.yaml`: planned EPOR-γ target configuration.

The four context fields have deliberately different meanings:

- `configured_max_context`: construction/configuration ceiling.
- `trained_max_context`: longest sequence used in training.
- `validated_max_context`: longest sequence that passed the context suite.
- `operational_default_context`: conservative default supported by evidence.

The public-family files therefore start with zero trained, validated, and
operational context. Their hardware profiles are marked `planned`. Parameter
counts are implementation targets checked allocation-free on PyTorch's meta
device. EPOR-γ's declared core is a research target; the v0.0.1 dense reference
graph counts the full model as active until elastic extraction is implemented.
