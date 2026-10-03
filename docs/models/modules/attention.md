# Attention Layers

## About

The `d9d.module.block.attention` package provides softmax and linear attention layers. Softmax attention layers compute scaled dot-product attention (SDPA) with a pluggable backend.

## Softmax Attention

### Grouped-Query Attention

`GroupedQueryAttention` implements [Grouped-Query Attention](https://arxiv.org/abs/2305.13245). If `num_key_value_heads` equals `num_attention_heads`, it works as Multi-Head Attention. If `num_key_value_heads` is 1, it works as Multi-Query Attention.

*   Uses a pluggable [SDPA backend](#scaled-dot-product-attention-backends).
*   Uses [rotary position embedding](./positional.md), optionally on a part of each head (`rope_dim`).
*   Supports optional [QK Normalization](https://arxiv.org/abs/2302.05442).
*   Supports optional sigmoid output gating.

### Multi-Head Latent Attention

`MultiHeadLatentAttention` implements Multi-Head Latent Attention (MLA) from [DeepSeek-V2](https://arxiv.org/abs/2405.04434).

*   Uses a pluggable [SDPA backend](#scaled-dot-product-attention-backends).
*   Uses [rotary position embedding](./positional.md).
*   `v_head_dim` must not exceed `qk_nope_head_dim + qk_rope_head_dim`.

### Scaled Dot-Product Attention Backends

Attention layers pass the scaled dot-product computation to a backend. The available backends are:

| `kind`                | Library                                                          | Sliding window | Learnable sinks | Explicit mask | Extra                                 |
|:----------------------|:-----------------------------------------------------------------|:---------------|:----------------|:--------------|:--------------------------------------|
| `"flash_attention_4"` | [FlashAttention 4](https://github.com/Dao-AILab/flash-attention) | Yes            | Yes             | No            | `d9d[backend-sdpa-flash-attention-4]` |
| `"flash_attention_2"` | [FlashAttention 2](https://github.com/Dao-AILab/flash-attention) | Yes            | No              | No            | `d9d[backend-sdpa-flash-attention-2]` |
| `"torch"`             | PyTorch `scaled_dot_product_attention`                           | No             | No              | Yes           | -                                     |
| `"eager"`             | Plain PyTorch ops                                                | Yes            | Yes             | Yes           | -                                     |

The `"torch"` backend can restrict PyTorch to specific kernels (`MATH`, `FLASH_ATTENTION`, `EFFICIENT_ATTENTION`, `CUDNN_ATTENTION`). The `"eager"` backend has no extra dependencies. Use it as a fallback and as a correctness reference for the other backends.

### Backend Selection

A layer picks its backend in this order:

1.  The `sdpa_backend` configuration passed to the layer.
2.  The `D9D_BACKEND_AUTO_SDPA` environment variable.
3.  Auto-detection: the first installed backend that supports the layer, in the order of the table above.

`D9D_BACKEND_AUTO_SDPA` holds a backend configuration as JSON, keyed by the `kind` field:

```bash
# Force the eager backend.
export D9D_BACKEND_AUTO_SDPA='{"kind": "eager"}'

# Force the PyTorch backend and restrict it to the FlashAttention kernel.
export D9D_BACKEND_AUTO_SDPA='{"kind": "torch", "backends": ["FLASH_ATTENTION"]}'
```

## Linear Attention

### Gated DeltaNet

`GatedDeltaNet` implements [Gated DeltaNet (GDN)](https://arxiv.org/abs/2412.06464). It is a linear attention layer. It combines the delta rule with Mamba-style data-dependent gating and short causal convolutions.

It uses kernels from [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) and requires the `d9d[linear-attention]` extra.

## Usage

`GroupedQueryAttention` takes RoPE embeddings from a `RotaryEmbeddingProvider`. Modules use late initialization, so call `reset_parameters()` before use.

```python
import torch

from d9d.module.block.attention import GroupedQueryAttention
from d9d.module.block.attention.sdpa import TorchSdpaBackendConfig
from d9d.module.block.positional import RotaryEmbeddingProvider, RotaryEmbeddingStyle

rope = RotaryEmbeddingProvider(
    rope_base=1_000_000,
    head_dim=128,
    max_position_ids=4096,
    style=RotaryEmbeddingStyle.HALF,
).cuda()
attention = GroupedQueryAttention(
    hidden_size=2048,
    num_attention_heads=32,
    num_key_value_heads=4,
    head_dim=128,
    qk_norm_eps=1e-6,
    is_causal=True,
    rope_style=RotaryEmbeddingStyle.HALF,
    sdpa_backend=TorchSdpaBackendConfig(),  # Omit to auto-detect
).cuda()
rope.reset_parameters()
attention.reset_parameters()

hidden_states = torch.randn(2, 16, 2048, device="cuda")
position_ids = torch.arange(16, device="cuda").expand(2, -1)
output = attention(hidden_states, attention_mask=None, position_embeddings=rope(position_ids))
```

## API Reference

::: d9d.module.block.attention

::: d9d.module.block.attention.sdpa

::: d9d.module.block.attention.linear
