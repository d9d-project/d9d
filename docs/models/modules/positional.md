# Positional Embeddings

## About

The `d9d.module.block.positional` package manages positional encoding logic.

## Features

### Rotary Positional Encoding

[Rotary Positional Encoding](https://arxiv.org/abs/2104.09864) from RoFormer.

See `RotaryEmbeddingProvider` and `RotaryEmbeddingApplicator` classes. 

First one is typically bound to a model class and is used for providing (cos, sin) embedding tensors for specified position IDs.

Second one is typically bound to attention module implementation and is used for modifying query and key states in runtime.

#### Embedding Layout Styles

The package supports multiple internal memory layouts for RoPE operations via the `RotaryEmbeddingStyle` enumeration. It is critical that both the provider and applicator share the identical style configuration:

::: d9d.module.block.positional
