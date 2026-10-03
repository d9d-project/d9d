# Positional Embeddings

## About

The `d9d.module.block.positional` package implements [rotary position embedding](https://arxiv.org/abs/2104.09864) (RoPE) from RoFormer.

## Rotary Position Embedding

RoPE uses two classes:

*   `RotaryEmbeddingProvider` usually belongs to the model. It caches the `(cos, sin)` embedding tensors and returns them for given position IDs.
*   `RotaryEmbeddingApplicator` usually belongs to the attention layer. It rotates the query and key states with these tensors.

### Layout Styles

`RotaryEmbeddingStyle` selects how features are paired for rotation:

*   `HALF` pairs the first half of the head dimension with the second half.
*   `INTERLEAVED` pairs adjacent features.

The provider and the applicator must use the same style.

### Scaling

`RotaryEmbeddingProvider` takes an optional `RopeScaling` strategy to extend the context length:

*   `NoRopeScaling` is the default and applies no scaling.
*   `LinearRopeScaling` divides all frequencies by a constant factor.
*   `NtkRopeScaling` implements [NTK-aware scaling](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/).
*   `YarnRopeScaling` implements [YaRN](https://arxiv.org/abs/2309.00071).

## Usage

```python
import torch

from d9d.module.block.positional import (
    RotaryEmbeddingApplicator,
    RotaryEmbeddingProvider,
    RotaryEmbeddingStyle,
    YarnRopeScaling,
)

provider = RotaryEmbeddingProvider(
    rope_base=10_000,
    head_dim=128,
    max_position_ids=16384,
    style=RotaryEmbeddingStyle.HALF,
    rope_scaling=YarnRopeScaling(
        factor=4.0, beta_fast=32.0, beta_slow=1.0, original_max_position_embeddings=4096
    ),
)
provider.reset_parameters()
applicator = RotaryEmbeddingApplicator(style=RotaryEmbeddingStyle.HALF)

position_ids = torch.arange(16).expand(2, -1)
cos, sin = provider(position_ids)  # (2, 16, 128) each

# Shapes: (batch, seq_len, num_heads, head_dim) and (batch, seq_len, num_kv_heads, head_dim).
query_states = torch.randn(2, 16, 32, 128)
key_states = torch.randn(2, 16, 4, 128)
query_states, key_states = applicator(query_states, key_states, cos, sin)
```

## API Reference

::: d9d.module.block.positional
