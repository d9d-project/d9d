# Attention Layers

## About

The `d9d.module.block.attention` package provides optimized attention mechanism implementations.

## Softmax Attention

### Grouped-Query Attention

`GroupedQueryAttention` is a [Grouped-Query Attention](https://arxiv.org/pdf/2305.13245) implementation.

Due to its abstract nature it is also can be used as a Multi-Head Attention and Multi-Query Attention module.

* Uses the `FlashSDPA` kernel for computation.
* Uses [Rotary Positional Encoding](./positional.md).
* Supports optional [QK Normalization](https://arxiv.org/pdf/2302.05442).
* Supports attention sinking.
* Supports sliding window.

### Multi-Head Latent Attention

`MultiHeadLatentAttention` is an implementation of the Multi-Head Latent Attention (MLA) mechanism introduced in [DeepSeek-V2](https://arxiv.org/abs/2405.04434).

* Uses the `FlashSDPA` kernel for computation.
* Uses [Rotary Positional Encoding](./positional.md).

### Scaled Dot-Product Attention Kernels

* `FlashSdpa` - FlashAttention 4.

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
