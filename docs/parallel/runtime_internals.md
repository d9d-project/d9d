# Parameter Runtime Internals

## About

This page explains how the [`ParameterRuntime`](./runtime.md) gathers weights, reduces gradients and bounds its memory. It is for readers who change the runtime or the pipeline program that drives it.

## Lifecycle

1.  `register_stage()` adds each stage module. It is allowed only while the runtime is unbound, once per stage.
2.  `bind()` discovers the marked groups and the stage remainder, plans buckets by group membership and allocates their buffers.
3.  Each step runs the regions inside `group_weights()`, reduces the buckets of `reduction_plan()`, and calls `wait()`, `scale_grads()` and `zero_grad()`.
4.  `unbind()` releases the buffers. A runtime can be bound again, e.g. after an offload.

The step methods raise `ValueError` on an unbound runtime. `invalidate_weights()`, `zero_grad()` and `unbind()` also work on an unbound runtime, because a checkpoint load or a teardown can come first.

`bind()` adds each stage to the bound state before it creates the buckets of the stage. If a bucket or a hook fails, `unbind()` releases what exists, and the runtime can be bound again.

`wait()` finishes the gradients once per step. A second call raises `ValueError`, also after a hook raised, so no hook runs twice. `zero_grad()` starts the next step. A missing reduction or a replaced gradient raises before any hook runs, so the caller can fix it and call `wait()` again.

## Sharded and Unsharded Parameters

Planning describes each parameter with a `ParameterPlan`: its owners, the sharded parameter, whether it trains, the compute dtype and the gradient dtype. It allocates nothing. `bind()` creates one `ParameterHandle` per plan, which adds the unsharded parameter. So each parameter has two `nn.Parameter` objects:

*   **Sharded**: The parameter that the module, the optimizer and checkpoints see. It is a `DTensor`, or a plain tensor outside a device mesh. It never changes identity.
*   **Unsharded**: The local tensor that regions compute with. Autograd writes its gradient.

Around each region, `group_weights()` writes the unsharded parameters into `module._parameters` and restores the sharded ones on exit. A tied parameter has one handle with several owners, so all of them get the same unsharded tensor and autograd adds up their gradients.

With `compute_dtype`, `bind()` also converts the non-persistent floating-point buffers of each stage, such as rotary embedding tables, to the compute dtype. Persistent buffers are training state, such as the expert bias of a router, so they keep their dtype. `unbind()` does not convert the buffers back.

A parameter without a gather uses a view of its local shard as the unsharded tensor, so optimizer writes reach it directly. With `compute_dtype`, it gets its own copy, which the first region of a step refreshes. The unsharded parameter of a gathered parameter views the weight buffer of its bucket, and the bucket creates it.

## Buckets

The runtime groups parameters into buckets. A `GatheredBucket` holds parameters with a gather and uses one all-gather and one reduce-scatter. A `ReplicatedBucket` holds trainable parameters without a gather, and autograd accumulates into its buffer directly. Both keep their gradients in a `GradientShard`. Many small collectives cost more launches and reach less bandwidth than one large one.

Parameters share a bucket when they share the mesh, the axis roles, the gradient dtype, the compute dtype, whether they train, and, for a gathered parameter, the groups that use it. A gathered bucket therefore holds the parameters of one set of groups, e.g. one decoder layer or the tied embedding and LM head. Buckets without a gather fill up to `bucket_size_mib`.

Bucket registration order depends only on the model. The actual forward trace determines gradient preparation and reduction points, including repeated uses.

## Buffers

The unsharded weights of a gathered bucket are views of one flat buffer. The runtime frees the memory with `untyped_storage().resize_(0)` and takes it back with `resize_()`. The tensor objects stay, so the views that autograd saved in the forward pass stay valid. Replacing `.data` would break them.

Uneven shards are padded to the size of the first one, as `torch.chunk` makes them, so the collectives need no uneven variants. All-gather places the shards of one rank next to each other, but a parameter needs its rows in order. One `split_with_sizes_copy()` per bucket converts between the layouts, as in FSDP2. The reduce-scatter converts back with one `cat()`.

A gathered bucket has three gradient memories:

*   **Gradient buffer**: The full gradient in the layout of the weights. Autograd writes it. It lives from the first region of the backward pass that writes it until its reduction.
*   **Gradient shard**: The shard of this rank after the reduce-scatter. It lives for the whole step, and the `.grad` of the sharded parameter views it. `scale_grads()` and `zero_grad()` change the shard, so this view is an invariant: after the post-accumulate-grad hooks, the runtime copies a replaced gradient back into the shard, and `wait()` checks the view. A parameter that no rank used in the step gets `.grad = None` until `zero_grad()`, so the optimizer skips it as without the runtime.
*   **Collective temporaries**: The packed input and the output of a collective.

A bucket without a gather has only a gradient shard of full size, and autograd accumulates into it directly.

## Memory Budget

The host queues work long before the GPU runs it, so memory that the host releases can still be read on another stream. `MemoryLedger` counts a released buffer as busy until the events of its last reads fire. `reserve()` waits on the oldest of them only when the budget is exhausted. `ManagedBuffer` pairs a flat tensor with the ledger and calls `record_stream()` for every stream that reads it.

The budget is a target with a floor: the runtime raises a smaller budget to the floor. The default budget is zero, so it keeps no weights between regions, as FSDP does. It counts gathered weights, gradient buffers and collective temporaries, but not the gradient shards, which live for the whole step. The floor covers the largest region: its gathered weights and the gradients that are live there. After observing a forward, the runtime updates the floor from the first and last occurrence of each bucket. This includes repeated groups and retained weight buffers. A factor of four leaves room for the prefetch of the next region and for collective buffers. The factor is a stopgap: the runtime does not yet measure the peak of a region.

## The Step

`group_weights()` runs these actions for one region:

1.  On the first region of a step, it refreshes the copies of `compute_dtype` and records the event after the optimizer writes.
2.  It finishes the gathers of the region. A gather waits for the optimizer event, so it never reads stale shards.
3.  It prepares the gradient buffers listed for this region in `ReductionPlan.before_region`. This is the last forward occurrence using each bucket.
4.  It starts the gathers of the next region, after its own ones, so the prefetch cannot take their budget.
5.  It puts the unsharded parameters into the modules and runs the block.
6.  It frees the weights of the buckets that the budget does not keep.

The pipeline program then reduces the buckets whose gradients are complete. `reduction_plan(stage, order, final)` lists preparation and reduction by region occurrence. A bucket reduces after its first forward occurrence, which ends its backward pass. A gathered bucket reduces after every pass that writes its gradients. The final pass also sums the gradient shards across the replicas and finds out which parameters any rank used: each rank marks a parameter when autograd writes its gradient, and the flags are reduced with `MAX` over every mesh dimension of the bucket. So a parameter that its layer shards, such as an expert, keeps a zero gradient on a rank whose part got no tokens. `wait()` checks that the program reduced every bucket in a final pass, reads the flags on the host, sets the gradients of unused parameters to `None` and runs the post-accumulate-grad hooks of the others. Reading the flags adds one host synchronization per step. When the budget is exhausted, the memory ledger also waits for retired buffers.

## Pipeline Program

`PipelineProgramCache` adds two things to the program of a rank:

*   **Windows**: In a distributed job without pipeline parallelism, it merges consecutive forward computes, and consecutive full backward computes, into `ForwardWindowComputeAction` and `BackwardWindowComputeAction`. With peers, a window would delay the output of a stage that another rank waits for.
*   **Final passes**: The cache records only whether each weight-writing action is final. The executor builds its `ReductionPlan` from the completed forward trace.

`WindowChain` holds the cut graph of one microbatch. Each boundary tensor has one producer. Every consumer adds to its gradient and runs before the producer in the backward pass, so the gradient is complete when the producer needs it. `forward_chains()` and `backward_chains()` run several chains region by region. Each microbatch has its own backward pass inside its compute context.

The first forward gathers groups on demand. Later forwards prefetch using the recorded order, or an explicit next-group hint. A window checks matching requests before gathering. Every participating rank must follow the same sequence. Returning from the outer coroutine creates an output graph that routes loss gradients to completed regions. Coroutine cleanup and each resume run inside the microbatch compute context.

## API Reference

::: d9d.parallel.runtime.memory

::: d9d.pipelining.stage.windows
