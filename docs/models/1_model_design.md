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

Please see [Pipelining API](./3_pipeline_parallelism.md).

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
