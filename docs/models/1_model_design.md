---
title: Model Design
---

# Model Design

## Bring Your own Model

d9d does **not** enforce you to use its model implementations. You are eager to use own custom 
implementations of any model you want, optionally using high-performant d9d's building blocks.

Just make sure to follow the main design principles described below.

## Main Principles

d9d opts for a "white-box" approach to modelling. We avoid heavy abstraction layers in favor of readable, standard PyTorch code.

### No LayerSpecs

Some distributed frameworks force users to define models via metadata specification objects to inject wrapping logic (like FSDP or Checkpointing) automatically. This makes debugging difficult.

In d9d, you write standard `nn.Module` classes. Use `nn.linear`, `nn.RMSNorm`, or d9d's optimized blocks directly. Distributed wrapping logic is handled transparently, maintaining the standard PyTorch look and feel.

### Explicit Composition

We avoid creating "Uber-Modules" - single, massive classes (e.g., `GenericTransformerBlock`) that handle every possible architectural variation (MoE, Dense, Post-Norm, Pre-Norm, Parallel Dense-Attention) via dozens of flags and parameters.

Instead, d9d promotes explicit composition like **HuggingFace Transformers** does. This composition makes the call stack distinct and the logic for a specific architecture easy to trace.

### Pipelining-Aware Models

**No Patching**

Most PyTorch models are written assuming a single device, with distributed support "patched in" later.

In d9d, while it is true for horizontal parallelism (distributing a model within a layer), splitting a model vertically for pipeline parallelism through patching could be problematic since **it disconnects the model definition from its runtime reality**.

Common approaches rely on the "Instantiate-then-Delete" pattern (see this [example](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)): creating a full model structure on a meta-device and then manually deleting attributes (`del model.layers[N:]`) based on the rank. We reject this pattern for these reasons:

1.  **Fragility and Implicit Coupling**: Externalizing the pipeline logic separates the definition of the model structure (the model class) from the definition of the model cut (the training script). This leads to fragile code where modifying the model architecture requires simultaneous, manual updates to the slicing logic in a completely different file. If the two drift apart, mistakes are often only caught at runtime.
2.  **Leaky Abstractions**: Patching attempts to hide pipeline logic from the model, but it often fails to do so cleanly. To support "Instantiate-then-Delete," module `forward` methods often become littered with `if self.layer is not None:` checks to handle potential voids. This makes the code confusing to read—the structure implies components exist, but the runtime logic implies they might not—creating a cognitive dissonance for developers trying to trace the data flow.
3.  **Structural Integrity**: In the patching paradigm, a model object exists in an invalid or "zombie" state until the external slicing script finishes its surgery. In d9d, we enforce **Construction Consistency**. When you initialize a `Model` with specific stage info, the object returned is compliant, complete, and valid immediately. It generates exactly the sub-graph required for that rank, with no dangling references or reduced-state post-processing required.

**Algorithmic Shape Inference**

Moreover, to enable efficient Pipeline Parallelism (PP), d9d needs to know the shapes of tensors passing between stages *before* execution begins to pre-allocate communication buffers and to support dynamic shapes between different pipeline calls.

Standard `torch.nn.Module`s do not expose this information. The `ModuleSupportsPipelining` protocol allows modules to implement functions enabling the framework to calculate pipeline stage input and output tensor shapes algorithmically without running a forward pass.

### Late Initialization

Constructing a large model on a single GPU (or even CPU RAM) often leads to immediate Out-Of-Memory (OOM) errors. `d9d` solves this via the `ModuleLateInit` protocol.

It is safe to use modules implementing this protocol with d9d's native [Trainer](TODO) framework. 

The Trainer will instantiate modules on the `meta` device (consuming no memory), lay out the distributed topology and sharding strategy. 

Only then `reset_parameters()` is called to materialize model weights without allocating unnecessary things.

## Reference Implementations

For reference implementations, please see [Qwen3-MoE](qwen3_moe.md).

::: d9d.module.base
    options:
        show_root_heading: true
        show_root_full_path: true
