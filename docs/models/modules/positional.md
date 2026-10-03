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

### Multimodal Rotary Positional Encoding (MRoPE)

`MultimodalRotaryEmbeddingProvider` consumes 3D position ids of shape `(3, batch, seq)` — separate
position planes for the temporal, height and width axes — and combines per-plane rotary
embeddings according to `mrope_section`. For text tokens, whose positions are identical across
the three planes, the output is numerically identical to the standard `RotaryEmbeddingProvider`.
The output plugs into the same `RotaryEmbeddingApplicator`.

See [Multimodality](../multimodality.md) for how the 3D position ids are computed.

::: d9d.module.block.positional
