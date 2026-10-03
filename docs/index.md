---
icon: lucide/house
---

# The d9d Project

**d9d** is a distributed training framework built on top of PyTorch 2.0. It aims to be hackable, modular, and efficient, designed to scale from single-GPU debugging to massive clusters running 6D-Parallelism.

## Installation

Install it with your package manager:

=== "pip"

    ```bash
    pip install d9d
    ```

=== "poetry"

    ```bash
    poetry add d9d
    ```

=== "uv"

    ```bash
    uv add d9d
    ```

### Extras

* `d9d[aim]`: [Aim](https://aimstack.io/) experiment tracker integration.
* `d9d[visualization]`: [Plotly](https://plotly.com/python/), for plotting learning rate schedules.
* `d9d[linear-attention]`: [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) kernels for linear attention.
* `d9d[backend-sdpa-flash-attention-2]`: [FlashAttention 2](https://github.com/Dao-AILab/flash-attention) SDPA backend kernels.
* `d9d[backend-sdpa-flash-attention-4]`: [FlashAttention 4](https://github.com/Dao-AILab/flash-attention) SDPA backend kernels.
* `d9d[moe]`: Mixture of Experts GPU kernels. You must build and install [DeepEP](https://github.com/deepseek-ai/DeepEP) and [grouped-gemm](https://github.com/fanshiqing/grouped_gemm/) manually first.
* `d9d[cce]`: Fused cross-entropy kernels. You must build and install [Cut Cross Entropy](https://github.com/apple/ml-cross-entropy) manually first.

## Documentation

Start with the [Table of Contents](./toc.md). You can read it from top to bottom.

## Examples

* **[Qwen3-MoE Pretraining](https://github.com/d9d-project/d9d/blob/main/example/qwen3_moe/pretrain.py):** causal LM pretraining of a Qwen3-MoE model.

---

## About

### Why another framework?

Distributed training frameworks such as **Megatron-LM** are monolithic: you run a script from the command line to train one of a set of *predefined* models, using *predefined* regimes. These systems are hard to hack and to integrate into new research workflows. They aim to be a complete end-to-end solution, which limits flexibility for experiment-driven research.

Writing your own distributed training solution from scratch is also hard. You must implement many low-level components that are the same across setups, such as distributed checkpoints and synchronization. You must also fix common performance bottlenecks yourself.

**d9d** fills the gap between monolithic frameworks and homebrew setups. It gives you modular building blocks for distributed training.

### What d9d is and isn't

In terms of **core concept**:

*   **IS** a pluggable framework for implementing distributed training regimes for your deep learning models.
*   **IS** built on clear interfaces and building blocks that may be composed and implemented in your own way.
*   **IS NOT** an all-in-one CLI platform for setting up pre-training and post-training like **torchtitan**, **Megatron-LM**, or **torchforge**.

In terms of **codebase & engineering**:

*   **IS** built on a **strong engineering foundation**: we enforce strict type checking and linting to catch errors before execution.
*   **IS** reliable: a suite of **over 450 tests** covers unit logic, integration flows, and end-to-end distributed scenarios.
*   **IS** eager to use performance hacks (like **DeepEP** or custom kernels) if they improve MFU, even if they aren't PyTorch-native.
*   **IS NOT** for legacy setups: we do not maintain backward compatibility with older PyTorch versions or hardware. We prefer simplicity and modern APIs (like `DTensor`).

### Key Philosophies

To balance hackability and performance, d9d follows these design principles:

*   **Composition over Monoliths**: we avoid "God Classes" like `DistributedDataParallel` or `ParallelDims` that own the entire execution loop. Instead, we provide composable and extendable APIs, such as per-layer horizontal parallelism strategies (`parallelize_replicate`, `parallelize_expert_parallel`, ...).
*   **White-Box Modelling**: we encourage standard PyTorch code. Models are not wrapped in metadata specifications. They are standard `nn.Module`s that implement small protocols.
*   **Pragmatic Efficiency**: we prefer native PyTorch, but we integrate non-native solutions if they improve MFU. For example, our MoE uses **DeepEP** communications, reindexing kernels from **Megatron-LM**, and grouped GEMM kernels.
*   **Graph-Based State Management**: our I/O system treats model checkpoints as directed acyclic graphs. You can transform architectures (e.g., merge `q`, `k`, `v` into `qkv`) on the fly while streaming from disk, without loading the whole checkpoint into memory.
*   **DTensors**: distributed parameters must be `torch.distributed.tensor.DTensor`s. DTensors know their topology, which makes checkpointing simpler. We use modern PyTorch 2.0 APIs (`DeviceMesh`) wherever we can.
