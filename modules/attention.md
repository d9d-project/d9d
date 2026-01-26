---
title: Attention Layers
---

# Attention Layers

## About

The `d9d.module.block.attention` package provides optimized attention mechanism implementations.

## Features

### Scaled Dot-Product Attention Kernels

* `FlashSdpa` - FlashAttention 2 (using new Torch SDPA API)

### Grouped-Query Attention

`GroupedQueryAttention` is a  [Grouped-Query Attention](https://arxiv.org/pdf/2305.13245) implementation.

Due to its abstract nature it is also can be used as Multi-Head Attention and Multi-Query Attention module.

Uses `FlashSDPA` kernel.

Uses [Rotary Positional Encoding](positional.md)

Supports optional [QK Normalization](https://arxiv.org/pdf/2302.05442).

::: d9d.module.block.attention
    options:
        show_root_heading: true
        show_root_full_path: true

::: d9d.module.block.attention.sdpa
    options:
        show_root_heading: true
        show_root_full_path: true
