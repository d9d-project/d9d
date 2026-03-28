# Qwen3 MoE

## About

The `d9d.module.model.qwen3_moe` package implements the Qwen3 Mixture-of-Experts model architecture.

The `d9d.module.parallelism.model.qwen3_moe` package implements default horizontal parallelism strategies for this model.

## HuggingFace Compatibility

d9d provides out-of-the-box support for streaming and converting HuggingFace checkpoints into the optimized d9d runtime format (and vice versa). 

These operations utilize the graph-based [State Mapping](../../model_states/mapper.md) engine. You may use the model state mappers provided for the [Model Provider](../../loop/interfaces/model.md) implementation.

::: d9d.module.model.qwen3_moe

::: d9d.module.parallelism.model.qwen3_moe
