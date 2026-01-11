---
title: Positional Embeddings
---

# Positional Embeddings

## About

The `d9d.module.block.positional` package manages positional encoding logic.

## Features

### Rotary Positional Encoding

[Rotary Positional Encoding](https://arxiv.org/abs/2104.09864) from RoFormer.

See `RotaryEmbeddingProvider` and `RotaryEmbeddingApplicator` classes. 

First one is typically bound to a model class and is used for providing (cos, sin) embedding tensors for specified position IDs.

Second one is typically bound to attention module implementation and is used for modifying query and key states in runtime.

::: d9d.module.block.positional
    options:
        show_root_heading: true
        show_root_full_path: true
