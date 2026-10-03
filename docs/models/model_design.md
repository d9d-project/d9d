# Model Design

## About

d9d does not require you to use its model implementations. You can bring your own model and optionally build it from d9d blocks. Your model must follow the principles on this page to work with the d9d training loop.

## Principles

d9d uses a "white-box" approach to modeling. Models are plain, readable PyTorch code without heavy abstraction layers.

### No Layer Specs

Some distributed frameworks make you describe a model with metadata objects. The framework then injects wrapping logic, such as FSDP or activation checkpointing. This makes debugging difficult.

In d9d, you write standard `nn.Module` classes. You can use `nn.Linear`, `nn.RMSNorm` or d9d blocks directly. You apply distributed strategies to submodules after construction, so the model code stays standard PyTorch.

### Explicit Composition

d9d avoids "uber-modules": large classes such as a `GenericTransformerBlock` that cover every architecture variant through dozens of flags. Examples of such variants are MoE vs. dense, pre-norm vs. post-norm and parallel attention.

Instead, d9d composes each architecture explicitly, as **Hugging Face Transformers** does. The call stack of each model is distinct, so its logic is easy to trace.

### Pipelining-Aware Models

See [Pipeline Parallelism](./pipeline_parallelism.md).

### Late Initialization

Building a large model on a single GPU, or even in CPU RAM, often runs out of memory. d9d avoids this with the `ModuleLateInit` protocol. Every model stage that you pass to the [`Trainer`](../loop/train.md) must implement it.

The `Trainer` initializes a model stage in this order:

1.  Construct the model on the `meta` device, which allocates no memory.
2.  Apply the horizontal parallelism strategy.
3.  Allocate empty storage for the local shards on the target device.
4.  Call `reset_parameters()` to initialize the weights.
5.  Load the source checkpoint, if one is configured.

## Reference Implementations

See [Qwen3 MoE](./model_catalogue/qwen3_moe.md) for a reference implementation.

## API Reference

::: d9d.module.base
