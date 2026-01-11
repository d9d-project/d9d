---
title: Mixture of Experts (MoE)
---

# Mixture of Experts (MoE)

## About

The `d9d.module.block.moe` package provides a complete, high-performance implementation of Sparse Mixture-of-Experts layers.

## Expert Parallelism

For information on setting up Expert Parallelism, see [this page](TODO)

## Features

### Sparse Expert Router

`TopKRouter` is a learnable router implementation.

It computes routing probabilities in FP32 to ensure numeric stability.

### Sparse Expert Token Dispatcher

`ExpertCommunicationHandler` is the messaging layer.

`NoCommunicationHandler` is used by default for single-GPU or Tensor Parallel setups where no token movement is needed.

`DeepEpCommunicationHandler` is enabled if using Expert Parallelism. It uses the [DeepEP](https://github.com/deepseek-ai/DeepEP) library for highly optimized all-to-all communication over NVLink/RDMA, enabling scaling to thousands of experts.

### Sparse Experts

`GroupedSwiGLU` provides a sparse SwiGLU experts module implementation.

Instead of looping over experts, it uses [Grouped GEMM](https://github.com/fanshiqing/grouped_gemm/) kernels to execute all experts in parallel, regardless of how many tokens each expert received.

### Shared Experts

Currently not supported, feel free to contribute :)

::: d9d.module.block.moe
    options:
        show_root_heading: true
        show_root_full_path: true

::: d9d.module.block.moe.communications
    options:
        show_root_heading: true
        show_root_full_path: true
