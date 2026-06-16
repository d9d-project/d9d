# Attention Layers

## About

The `d9d.module.block.attention` package provides optimized attention mechanism implementations.

## Softmax Attention

### Grouped-Query Attention

`GroupedQueryAttention` is a [Grouped-Query Attention](https://arxiv.org/pdf/2305.13245) implementation.

Due to its abstract nature it is also can be used as a Multi-Head Attention and Multi-Query Attention module.

* Uses a pluggable [SDPA backend](#scaled-dot-product-attention-backends) for computation.
* Uses [Rotary Positional Encoding](./positional.md).
* Supports optional [QK Normalization](https://arxiv.org/pdf/2302.05442).
* Supports attention sinking.
* Supports sliding window.

### Multi-Head Latent Attention

`MultiHeadLatentAttention` is an implementation of the Multi-Head Latent Attention (MLA) mechanism introduced in [DeepSeek-V2](https://arxiv.org/abs/2405.04434).

* Uses a pluggable [SDPA backend](#scaled-dot-product-attention-backends) for computation.
* Uses [Rotary Positional Encoding](./positional.md).

### Scaled Dot-Product Attention Backends

The attention modules delegate the core scaled dot-product computation to a
pluggable backend. Available backends are:

* `"flash_attention_4"` - [FlashAttention 4](https://github.com/Dao-AILab/flash-attention).
  Supports sliding windows and learnable sinks; does not accept explicit masks.
* `"flash_attention_2"` - [FlashAttention 2](https://github.com/Dao-AILab/flash-attention).
  Supports sliding windows but not sinks or explicit masks.
* `"torch"` - PyTorch's fused `scaled_dot_product_attention`. Optionally pins a
  specific `SDPBackend` (`MATH`, `FLASH_ATTENTION`, `EFFICIENT_ATTENTION`,
  `CUDNN_ATTENTION`). Does not support sinks or sliding windows.
* `"eager"` - a portable, dependency-free pure-PyTorch reference. Supports
  GQA/MQA, causal masking, sliding windows, explicit attention masks, and
  learnable attention sinks. Primarily useful as a fallback and as a correctness
  reference for the other backends.

The backend is chosen via `build_sdpa_backend()`. When no explicit configuration
is given, it is auto-detected based on installed kernels and the structural
requirements (sinks, windows, explicit attention masks) of the layer.
The auto-detected choice can be overridden through the
`D9D_BACKEND_AUTO_SDPA` environment variable (a JSON backend configuration).

The JSON is the serialized form of the corresponding backend config, keyed by a
`kind` discriminator. For example:

```bash
# Force the pure-PyTorch eager backend.
export D9D_BACKEND_AUTO_SDPA='{"kind": "eager"}'

# Force PyTorch's fused SDPA, pinning a specific kernel.
export D9D_BACKEND_AUTO_SDPA='{"kind": "torch", "backends": ["FLASH_ATTENTION"]}'
```

This override only affects auto-detection; passing an explicit `sdpa_backend`
configuration to a layer takes precedence over the environment variable.

::: d9d.module.block.attention
    options:
      heading_level: 3

::: d9d.module.block.attention.sdpa
    options:
      heading_level: 3

## Linear Attention

### Gated DeltaNet

`GatedDeltaNet` is an implementation of the [Gated DeltaNet (GDN)](https://arxiv.org/abs/2412.06464) attention mechanism. 

It acts as a linear attention alternative that combines the Delta Rule with Mamba-style data-dependent gating and short causal convolutions.

::: d9d.module.block.attention.linear
    options:
      heading_level: 3
