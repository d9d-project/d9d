# Gradient Synchronization

## About

The `d9d.internals.grad_sync` package synchronizes gradients of `DTensor` parameters in distributed training.

PyTorch `DistributedDataParallel` applies one communication strategy to the whole model. This package instead supports the mixed layouts of ND parallelism, which combines data, tensor, context and pipeline parallelism. It reads the `DTensor` placements of each parameter to find the mesh dimensions that need an all-reduce. Then it groups the parameters into communication buckets.

!!! warning "Internal API"
    If you use the standard d9d training loop, you do not need to call this package. d9d synchronizes the gradients itself. This page is for users who extend d9d.

## Bucketing and Flattening

When many small tensors are reduced, latency dominates the communication cost. So `GradientSynchronizer` groups parameters into **buckets**.

A `SyncGradientBucket` stores the gradients of all its parameters in one contiguous buffer. Its reduction is a single `all_reduce` per process group on this buffer, instead of hundreds of small operations.

Parameters share a bucket only if they have the same:

1.  **Optimizer parameter group**
2.  **Device**
3.  **Gradient dtype**
4.  **Reduce mesh**: the mesh dimensions where the parameter has a `Replicate` placement

The `bucket_size_mb` argument limits the size of a bucket in MiB. Parameters are bucketed in reverse order, because the backward pass produces gradients roughly in that order. Parameters with no `Replicate` placement need no reduction. They go to local buckets that do not communicate.

## Asynchronous Reduction

Training often accumulates gradients over several microbatches before an optimizer step. This package manages the `DTensor` gradients during the accumulation. You do not need a `no_sync` context manager.

1.  **Local accumulation**: During the backward pass of the first $N-1$ microbatches, local gradients accumulate in the bucket buffer. The parameter `DTensor` is `Replicate`, so the gradient also has a `Replicate` placement across the data parallel mesh. Its data still differs between ranks at this point.

2.  **Automatic trigger**: Each bucket counts the gradient accumulations of its parameters. The `all_reduce` starts *only* when all parameters of the bucket reach the `require_accumulations` count. The trigger runs inside the backward hook of the *last* microbatch. So the communication overlaps with the backward pass of the remaining layers. The communication runs on a **separate CUDA stream**. You **must** wait for it before you use the gradients on your default stream.

3.  **Synchronization**: When the asynchronous reduction completes, the flat buffer holds the globally summed gradient. The gradients have the same `Replicate` placement as their parameters. So the optimizer can use them without any further synchronization.

## API Reference

::: d9d.internals.grad_sync
