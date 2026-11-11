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

Instead, d9d composes each architecture explicitly, as Hugging Face Transformers does. Each model builds its own layers, so its logic is easy to trace. Qwen3 Dense and Qwen3 MoE expose their execution boundaries through [resumable forwards](../pipelining/index.md#resumable-forwards).

### Pipelining-Aware Models

See [Pipeline Parallelism](../pipelining/index.md).

### Late Initialization

Building a large model on a single GPU, or even in CPU RAM, often runs out of memory. d9d avoids this with the `ModuleLateInit` protocol. Every model stage that you pass to the [`Trainer`](../loop/train.md) must implement it.

The `Trainer` initializes a model stage in this order:

1.  Construct the model on the `meta` device, which allocates no memory.
2.  Place the parameters with the horizontal parallelism strategy.
3.  Allocate empty storage for the local shards on the target device.
4.  Call `reset_parameters()` to initialize the weights.
5.  Load the source checkpoint, if one is configured.

## Parameter Groups

The [Parameter Runtime](../parallel/runtime.md) gathers the weights of each parameter group before executing its region. Mark groups with `mark_window()` and express execution order in one `forward_steps()`. A model without this protocol runs as one whole-stage group.

Use `await steps.call(module, ...)` for module calls. Use `state = await steps.enter(module, state)` for arbitrary code inside a group, and `await steps.finish(state)` to close it. See [Resumable Forwards](../pipelining/index.md#resumable-forwards) for complete examples of both forms.

*   **Group membership**: `mark_window()` returns the same module without wrappers or execution hooks. A containing group excludes marked descendants. Tied parameters keep one handle and belong to each group that uses them.
*   **Live state**: Expose every new differentiable value needed later at a manual boundary. Reassign the returned state, including residuals and collected hidden states. References outside that state are not rewritten.
*   **Completed values**: Results of completed calls keep their producers. Local variables can hold them for later calls and regions.
*   **Stable order**: Every microbatch and participating rank uses the same group sequence while the runtime is bound. Repeated groups are allowed. Every parameter must be covered by an executed group.
*   **Compute contexts**: Contexts such as `no_grad()` and autocast must not span a boundary. The executor raises an error if grad mode, inference mode or autocast changes across one. It enters the microbatch context around each resume.
*   **Checkpointing**: Checkpoint synchronous computations inside a region. A coroutine cannot replay its earlier regions.
*   **Tied weights**: Keep tied parameters on one pipeline stage. Multiple groups and repeated calls can use the same parameter.

The Qwen3 backbones expose embedding, decoder-layer and normalization groups in `forward_steps()`. Snapshot collection and checkpointing remain in that function. `DecoderWithHead` and `DecoderWithHeads` compose the backbone and their heads through the same protocol.

## Reference Implementations

See [Qwen3 MoE](./qwen3_moe.md) for a reference implementation.

## API Reference

::: d9d.nn
