# Parameter Runtime Internals

## About

This page explains how the [`ParameterRuntime`](./runtime.md) gathers weights, reduces gradients and bounds its memory. It is for readers who change the runtime or the pipeline program that drives it.

## Sharded and Unsharded Parameters

Each parameter has two `nn.Parameter` objects, held in a `ParameterHandle`:

*   **Sharded**: The parameter that the module, the optimizer and checkpoints see. It is a `DTensor`, or a plain tensor outside a device mesh. It never changes identity.
*   **Unsharded**: The local tensor that segments compute with. Autograd writes its gradient.

Around each segment, `segment_weights()` writes the unsharded parameters into `module._parameters` and restores the sharded ones on exit. A tied parameter has one handle with several owners, so all of them get the same unsharded tensor and autograd adds up their gradients.

A parameter without a gather uses a view of its local shard as the unsharded tensor, so optimizer writes reach it directly. With `compute_dtype`, it gets its own copy, which the first segment of a step refreshes. Planning gives such parameters a tensor on the `meta` device, and the copy is made after the gathered buckets bind, so a gathered parameter never allocates one.

## Buckets

The runtime groups parameters into buckets. One bucket uses one all-gather and one reduce-scatter. A collective per parameter is small enough that NCCL picks its low-latency protocol, which loses most of the bandwidth between nodes.

Parameters share a bucket when they share the mesh, the axis roles, the gradient dtype, the compute dtype, whether they train, and, for a gathered parameter, the segments that use it. A gathered bucket therefore holds the parameters of one set of segments, e.g. one decoder layer or the tied embedding and LM head. Buckets without a gather fill up to `bucket_size_mb`.

The runtime plans buckets in backward order, segment by segment from the last one. The order depends only on the model, so the collectives match on every rank.

## Buffers

The unsharded weights of a gathered bucket are views of one flat buffer. The runtime frees the memory with `untyped_storage().resize_(0)` and takes it back with `resize_()`. The tensor objects stay, so the views that autograd saved in the forward pass stay valid. Replacing `.data` would break them.

Uneven shards are padded to the size of the first one, as `torch.chunk` makes them, so the collectives need no uneven variants. All-gather places the shards of one rank next to each other, but a parameter needs its rows in order. One `split_with_sizes_copy()` per bucket converts between the layouts, as in FSDP2. The reduce-scatter converts back with one `cat()`.

A gathered bucket has three gradient memories:

*   **Gradient buffer**: The full gradient in the layout of the weights. Autograd writes it. It lives from the first segment of the backward pass that writes it until its reduction.
*   **Gradient shard**: The shard of this rank after the reduce-scatter. It lives for the whole step, and the `.grad` of the sharded parameter views it. `scale_grads()` and `zero_grad()` change the shard, so this view is an invariant: after the post-accumulate-grad hooks, the runtime copies a replaced gradient back into the shard, and `wait()` checks the view.
*   **Collective temporaries**: The packed input and the output of a collective.

A bucket without a gather has only a gradient shard of full size, and autograd accumulates into it directly.

## Memory Budget

The host queues work long before the GPU runs it, so memory that the host releases can still be read on another stream. `MemoryLedger` counts a released buffer as busy until the events of its last reads fire. `reserve()` waits on the oldest of them only when the budget is exhausted. `ManagedBuffer` pairs a flat tensor with the ledger and calls `record_stream()` for every stream that reads it.

The minimum budget covers the largest segment: its gathered weights and the gradients that are live there. Tied gradients stay live from their last segment to their first one. A rough factor of four leaves room for the prefetch of the next segment and for collective buffers.

## The Step

`segment_weights()` runs these actions for one segment:

1.  On the first segment of a step, it refreshes the copies of `compute_dtype` and records the event after the optimizer writes.
2.  It finishes the gathers of the segment. A gather waits for the optimizer event, so it never reads stale shards.
3.  In a pass that writes gradients, it clears the gradient buffer of each bucket whose last segment this is. A tied bucket writes its gradient over several segments, from the last one on.
4.  It starts the gathers of the next segment, after its own ones, so the prefetch cannot take their budget.
5.  It puts the unsharded parameters into the modules and runs the block.
6.  It frees the weights of the buckets that the budget does not keep.

The pipeline program then reduces the buckets whose gradients are complete. `reduction_plan()` lists them by segment: a bucket reduces after its first segment, which ends its backward pass. A gathered bucket reduces after every pass that writes its gradients. The final pass also sums the gradient shards across the replicas and runs the post-accumulate-grad hooks. `wait()` checks that the program reduced every bucket in a final pass.

## Pipeline Program

`PipelineProgramCache` adds two things to the program of a rank:

*   **Windows**: In a distributed job without pipeline parallelism, it merges consecutive forward computes, and consecutive full backward computes, into `ForwardWindowComputeAction` and `BackwardWindowComputeAction`. With peers, a window would delay the output of a stage that another rank waits for.
*   **Reductions**: For every action that writes weight gradients, it stores the `ReductionPlan` of its stage. Only the last such action of a stage is final.

`SegmentChain` holds the cut graph of one microbatch. Each boundary tensor has one producer. Every consumer adds to its gradient and runs before the producer in the backward pass, so the gradient is complete when the producer needs it. `forward_chains()` and `backward_chains()` run several chains segment by segment, with one autograd pass per segment for the whole window.

## API Reference

::: d9d.parallel.memory

::: d9d.pipelining.stage.segments
