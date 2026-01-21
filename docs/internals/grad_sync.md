---
title: Gradient Synchronization
---

# Gradient Synchronization

## About

!!! warning "Warning:" 
    If you are utilizing the standard `d9d` training infrastructure, you **do not** need to call these functions manually. The framework automatically handles gradient synchronization. This package is primarily intended for users extending `d9d`.

The `d9d.internals.grad_sync` package provides low-level primitives for synchronizing gradients in distributed training setups utilizing `DTensor`.

Unlike standard PyTorch `DistributedDataParallel` which assumes a uniform communication strategy for the entire model, this package is designed to work with heterogeneous distributions often found in ND-parallelism (e.g., mixtures of Data, Tensor, Sequence, and Pipeline parallelism). It inspects `DTensor` placements to automatically determine which dimensions require reduction (all-reduce) and groups parameters into efficient communication buckets.

## Core Concepts

### Bucketing & Flattening

Communication overhead is dominated by latency when reducing many small tensors. To mitigate this, `GradientSynchronizer` groups parameters into **Buckets**.

Inside a `SyncGradientBucket`, gradients for multiple parameters are flattened into a single contiguous block of memory. When a reduction is triggered, the system performs a single `all_reduce` operation on this large buffer instead of hundreds of small operations.

Parameters are grouped automatically based on:

1.  **Device**
2.  **DType**
3.  **Associated DeviceMesh**

### Asynchronous Reduction

In large-scale training, effective batch size is often increased by accumulating gradients over multiple micro-batches before performing an optimizer step. This package manages the lifecycle of distributed `DTensor` gradients during this accumulation phase without simple `no_sync` context managers.

1.  **Local Accumulation**: 
    During the backward pass of the first $N-1$ micro-batches, local gradients are accumulated into the bucket's buffer. Conceptually, while the parameter `DTensor` is `Replicate`d, these intermediate local gradients also represent a `Replicate` (although contain different data) state across the Data Parallel mesh.

2.  **Automatic Triggering**: 
    Each bucket maintains an internal counter. The `all_reduce` communication is *only* triggered when the specific parameter group has reached the `require_accumulations` count. This trigger happens automatically inside the backward hook of the *last* micro-batch, allowing communication to immediately overlap with the computation of remaining layers higher up in the model.

3.  **Synchronization**: 
    Once the asynchronous reduction completes, the flat buffer contains the globally summed gradient. Metadata of the contained parameter gradients is marked as `Replicate`, making them safe for the Optimizer to consume without involving synchronization later.

::: d9d.internals.grad_sync
    options:
        show_root_heading: true
        show_root_full_path: true
