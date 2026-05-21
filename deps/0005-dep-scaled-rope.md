---
DEP: 0005
Title: Scaled RoPE
Author: Illarion Iov @AngrySigma
Status: Implemented
Type: Feature
Created: 2026-04-29
---

# DEP-0005: Scaled RoPE

## Abstract
This proposal extends `RotaryEmbeddingProvider` with an optional `rope_scaling` param to control RoPE inverse frequencies computation and cos/sin tables post-scale.
Along with the base class, several prominent implementations are added.
Base class `RopeScaling` is an interface that acts as the extension point for adopting long-context scaling strategies.

## Motivation
Many long-context LLMs ship with non-default RoPE configurations. For instance, GPT-OSS uses YaRN.
Current `RotaryEmbeddingProvider` always uses the standard geometric-progression inverse frequencies, making these checkpoints either wrong in train and inference, or significantly diverging.
Generic opt-in scaling extension points let any model that constructs a `RotaryEmbeddingProvider` adopt long-context RoPE as a config change seamlessly, including dynamic scaling that adjusts per-iteration based on sequence length.


## Design Proposal

A new abstract base class in `d9d/module/block/positional/rope_scaling.py`:

```python
class RopeScaling(ABC):
    @abstractmethod
    def inverse_frequencies(self, rope_base: int, head_dim: int) -> torch.Tensor: ...

    def attention_mscale(self) -> float:
        return 1.0
```

Six concrete subclasses:

- `NoRopeScaling` — the default implementation; returns standard geometric-progression inverse frequencies with no mscale. Used internally when `rope_scaling=None` is passed.
- `LinearRopeScaling(factor)` — divides the standard inverse frequencies by `factor`. `attention_mscale` stays at the default `1.0`.
- `YarnRopeScaling(factor, beta_fast, beta_slow, original_max_position_embeddings)` — wavelength-bin ramp between extrapolation and interpolation per dimension; `attention_mscale = 0.1 * log(factor) + 1.0` for `factor > 1`.
- `NtkRopeScaling(factor)` — computes a new floating-point RoPE base mathematically to scale down low-frequency dimensions while preserving high-frequency dimensions.

`prepare_rotary_cos_sin_emb` and `RotaryEmbeddingProvider.__init__` gain a single optional kwarg: `rope_scaling: RopeScaling | None = None`. When `None`, it is resolved to `NoRopeScaling()` at entry — so existing call sites are byte-for-byte equivalent.

## Usage

```python
from d9d.module.block.positional import (
    RotaryEmbeddingProvider,
    RotaryEmbeddingStyle,
    YarnRopeScaling,
)

provider = RotaryEmbeddingProvider(
    rope_base=150_000,
    head_dim=64,
    max_position_ids=131_072,
    style=RotaryEmbeddingStyle.HALF,
    rope_scaling=YarnRopeScaling(
        factor=32.0,
        beta_fast=32.0,
        beta_slow=1.0,
        original_max_position_embeddings=4096,
    ),
)
```

## Backward Compatibility

Fully backward compatible.

## Alternatives Considered

A pydantic discriminated-union of plain dataclasses was considered, but the scaling config also owns the *computation* of inverse frequencies and the mscale.
Placing data and computation in a small ABC removes a separate dispatch inside `prepare_rotary_cos_sin_emb` and keeps the provider unaware of which scaling family is in use.
