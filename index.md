---
title: Home
---

# The d9d Project

**d9d** is a distributed training framework built on top of PyTorch 2.0. It aims to be hackable, modular, and efficient, designed to scale from single-GPU debugging to massive clusters running 6D-Parallelism.

## Installation

Just use your favourite package manager:
```bash
pip install d9d
poetry add d9d
uv add d9d
```

### Extras

* `d9d[aim]`: [Aim](https://aimstack.io/) experiment tracker integration.
* `d9d[visualization]`: Plotting libraries required to some advanced visualization functionality.
* `d9d[moe]`: Efficient Mixture of Experts GPU kernels. You should build and install some dependencies manually before installation: [DeepEP](https://github.com/deepseek-ai/DeepEP), [grouped-gemm](https://github.com/fanshiqing/grouped_gemm/).
* `d9d[cce]`: Efficient Fused Cross-Entropy kernels. You should build and install some dependencies manually before installation: [Cut Cross Entropy](https://github.com/apple/ml-cross-entropy).

## Why another framework?

Distributed training frameworks such as **Megatron-LM** are monolithic in the way you run a script from the command line to train any of a set of *predefined* models, using *predefined* regimes. While powerful, these systems can be difficult to hack and integrate into novel research workflows. Their focus is often on providing a complete, end-to-end solution, which can limit flexibility for experimentally-driven research.

Conversely, creating your own distributed training solution from scratch is tricky. You have to implement many low-level components (like distributed checkpoints and synchronization) that are identical across setups, and manually tackle common performance bottlenecks.

**d9d** was designed to fill the gap between monolithic frameworks and homebrew setups, providing a modular yet effective solution for distributed training.

## What d9d is and isn't

In terms of **core concept**:

*   **IS** a pluggable framework for implementing distributed training regimes for your deep learning models.
*   **IS** built on clear interfaces and building blocks that may be composed and implemented in your own way.
*   **IS NOT** an all-in-one CLI platform for setting up pre-training and post-training like **torchtitan**, **Megatron-LM**, or **torchforge**.

In terms of **codebase & engineering**:

*   **IS** built on a **strong engineering foundation**: We enforce strict type-checking and rigorous linting to catch errors before execution.
*   **IS** reliable: The framework is backed by a suite of **over 450 tests**, covering unit logic, integration flows, and End-to-End distributed scenarios.
*   **IS** eager to use performance hacks (like **DeepEp** or custom kernels) if they improve MFU, even if they aren't PyTorch-native.
*   **IS NOT** for legacy setups: We do not maintain backward compatibility with older PyTorch versions or hardware. We prioritize simplicity and modern APIs (like `DTensor`).

## Key Philosophies

To achieve the balance between hackability and performance, d9d adheres to specific design principles:

*   **Composition over Monoliths**: We avoid "God Classes" like `DistributedDataParallel` or `ParallelDims` that assume ownership of the entire execution loop. Instead, we provide composable and extendable APIs. For instance, specific horizontal parallelism strategies for specific layers (`parallelize_replicate`, `parallelize_expert_parallel`, ...).
*   **White-Box Modelling**: We encourage standard PyTorch code. Models are not wrapped in obscure metadata specifications; they are standard `nn.Module`s that implement lightweight protocols.
*   **Pragmatic Efficiency**: While we prefer native PyTorch, we are eager to integrate non-native solutions if they improve MFU. For example, we implement MoE using **DeepEp** communications, reindexing kernels from **Megatron-LM**, and efficient grouped-GEMM implementations.
*   **Graph-Based State Management**: Our IO system treats model checkpoints as directed acyclic graphs. This allows you to transform architectures (e.g., merging `q`, `k`, `v` into `qkv`) on-the-fly while streaming from disk, without massive memory overhead.
*   **DTensors**: We mandate that distributed parameters be represented as `torch.distributed.tensor.DTensor`. This simplifies checkpointing by making them topology-aware automatically. We leverage modern PyTorch 2.0 APIs (`DeviceMesh`) as much as possible.

---

## Documentation

Please navigate through our [Table of Contents](./toc.md) - you can safely read everything from top to bottom.

## Examples

### Qwen3-MoE Pretraining
An example showing causal LM pretraing for the Qwen3-MoE model.

WIP: MoE load balancing is currently work in progress.

[Link](https://github.com/d9d-project/d9d/blob/main/example/qwen3_moe/pretrain.py).
