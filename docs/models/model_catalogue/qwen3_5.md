# Qwen3.5 Dense

## About

The `d9d.module.model.qwen3_5` package implements the Qwen3.5 dense model architecture: a hybrid
decoder interleaving linear-attention (Gated DeltaNet) and full-attention layers (3:1 by
default), with a SwiGLU MLP in every layer, zero-centered RMSNorm, partial interleaved MRoPE, and
an optional vision encoder for image and video inputs (see
[Multimodality](../multimodality.md)).

Two task variants are provided:

* `Qwen3p5ForCausalLM` — text-only causal language modelling.
* `Qwen3p5ForConditionalGeneration` — multimodal (vision + text) causal language modelling. The
  vision encoder and the media-embedding merge live strictly on the first pipeline stage.

The `d9d.module.parallelism.model.qwen3_5` package implements default horizontal parallelism
strategies for this model.

## HuggingFace Compatibility

d9d provides out-of-the-box support for streaming and converting HuggingFace checkpoints into the
optimized d9d runtime format (and vice versa).

These operations utilize the graph-based [State Mapping](../../model_states/mapper.md) engine. You
may use the model state mappers provided for the
[Model Provider](../../loop/interfaces/model.md) implementation.

::: d9d.module.model.qwen3_5

::: d9d.module.parallelism.model.qwen3_5
