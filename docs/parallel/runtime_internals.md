# Parameter Runtime Internals

## About

The `ParameterRuntime` is data parallelism that its caller drives region by region. A region is a part of the forward pass that needs the parameters of one parameter group. Modules keep their sharded parameters. Around each region, the runtime puts the unsharded parameters of its group into the modules. After the backward pass of a group, it reduces the gradients that are complete. This page is for readers who change the runtime or write code that drives it.

## Lifecycle

The `ParameterRuntime` holds the configuration and the registered modules. What exists only while the modules are bound belongs to the `BoundParameters` that `bind()` returns:

1.  `ParameterRuntime.register()` adds a module and the roots of its parameter groups, once per module.
2.  `ParameterRuntime.bind()` plans the buckets, creates the unsharded parameters and the gradient buffers, and returns the `BoundParameters`.
3.  Each step starts with `BoundParameters.zero_grad()`, runs the regions of every microbatch, and ends with `BoundParameters.wait()`.
4.  `BoundParameters.close()` releases the buffers and the hooks, and sets the `.grad` of the sharded parameters to `None`. A second call does nothing. After it, `bind()` binds the modules again, e.g. after an offload.

As a context manager, `BoundParameters` calls `close()` on exit. Its methods raise `ValueError` after `close()`. While a `BoundParameters` is open, `register()` and a second `bind()` raise, because two binds would point the gradients of the sharded parameters at their own buffers. If `bind()` fails, it releases what it built, and the runtime can bind again.

A `ModuleParameters` object holds no memory after `close()`. Its `group_weights()`, `record_order()` and `backward()` then raise, so take it again with `parameters()` from each new `BoundParameters`.

## Parameter Groups

A parameter group is the set of parameters that a region needs. Its root is a module that `register()` receives. The group holds the parameters of the root and its descendants, except those of other roots. The registered module is the root of its own group, which holds the parameters outside every other group. A module without other roots is one group.

`register()` raises if a root is repeated, is the module itself or lies outside it.

A parameter that several groups hold is a tied parameter, e.g. a tied embedding and LM head. It belongs to every group that holds it.

## Sharded and Unsharded Parameters

Each parameter has two `nn.Parameter` objects:

*   **Sharded**: The parameter that the module, the optimizer and checkpoints see. It is a `DTensor`, or a plain tensor in a job without parallelism. It never changes identity.
*   **Unsharded**: The local tensor that regions compute with. Autograd writes its gradient.

Around each region, the runtime writes the unsharded parameters into `module._parameters` and restores the sharded ones on exit. A tied parameter has one unsharded parameter for all its owners, so autograd adds up their gradients.

The unsharded parameter of a gathered parameter views the weight buffer of its bucket. A parameter without a gather views its local tensor, so optimizer writes reach it directly.

The unsharded parameters keep the dtype of the sharded ones. The gradients use `param.grad_dtype`. A trainable parameter with `grad_dtype = None` raises in `bind()`, because a gradient buffer needs one dtype. Without training, the unsharded parameters require no gradients.

## Buckets

The runtime puts parameters into buckets, so that one collective serves many parameters. Many small collectives cost more launches and reach less bandwidth than one large one.

*   **Gathered bucket**: Parameters that data parallelism shards with `Shard(0)` along a dimension of `gather_dims`. Each use gathers the weights with one all-gather. Each backward pass reduces the gradients with one reduce-scatter.
*   **Replicated bucket**: Trainable parameters without a gather. Autograd accumulates into the gradient buffer of the bucket directly.

Parameters share a gathered bucket when they share the groups that hold them, so a gathered bucket holds e.g. one decoder layer. They must also share the mesh, the roles of the mesh dimensions, the dtypes and whether they train. Replicated buckets fill up to `bucket_size_mib`.

After the final backward pass of a step, every trainable bucket sums its gradients over the `Replicate` dimensions of the mesh. The runtime leaves a `Shard` along any other dimension to the layer, e.g. the experts of expert parallelism.

## Buffers

The unsharded weights of a gathered bucket are views of one weight buffer. The runtime frees its memory with `untyped_storage().resize_(0)` and takes it back with `resize_()`. The tensor objects stay, so the views that autograd saved in the forward pass stay valid. Replacing `.data` would break them.

`torch.chunk` makes the first chunk the largest, so the shard of each rank is padded to its size. So the collectives need no uneven variants. All-gather places the chunks of one rank next to each other, but a parameter needs its own chunks next to each other. One `split_with_sizes_copy()` per bucket converts between the layouts. The reduce-scatter converts back with one `cat()`.

A gathered bucket has a gradient buffer and a gradient shard:

*   **Gradient buffer**: The full gradient in the layout of the weights. Autograd writes it. It lives from the first region of a backward pass that writes it until the runtime frees it after the reduction.
*   **Gradient shard**: The shard of this rank after the reduce-scatter. It lives while the runtime is bound, and the `.grad` of the sharded parameter views it.

A replicated bucket has one gradient buffer of full size. The `.grad` of the sharded and of the unsharded parameter both view it.

`scale_grads()` and `zero_grad()` change the gradient shards and the replicated gradient buffers in place. So if a post-accumulate-grad hook replaces a gradient, `wait()` copies the new value back into the gradient shard or the replicated gradient buffer. `wait()` raises if a gradient was replaced between `zero_grad()` and `wait()`.

## Streams

A buffer belongs to the stream that writes it, and only that stream frees its memory. Before it frees a buffer, it waits for an event after the last read on another stream. So the memory stays bounded however far the host runs ahead:

*   **Weights**: The gather stream writes them and the compute stream reads them. Before the weights of a region are freed, the gather stream waits for the end of that region. So the gather of a group starts only after the compute of the group two regions before it. At most about two groups are gathered.
*   **Gradient buffer**: The compute stream writes it, and the reduce stream copies it into the layout of the reduce-scatter. A wait for that copy right away would stall the backward pass behind the reductions. So the compute stream frees a gradient buffer when the bucket two places later allocates its own. By then, the copy has usually ended.

## The Step

### Forward

The first forward pass after `bind()` records the order of the module with `record_order()`. The order holds the root of the group of each region, or `None` for a region without a group. Later forward passes must have the same order, which `record_order()` checks.

For each region, the caller enters `ModuleParameters.group_weights()`, which:

1.  Records an event on the compute stream. Every gather of the region waits for it, so it never reads stale shards.
2.  Gathers the weights of the group, unless a previous region prefetched or kept them.
3.  Starts the gather of the group of `next_root`. The caller can take it from `ModuleParameters.order`, which is `None` until the first `record_order()`.
4.  Puts the unsharded parameters into the modules and runs the region.
5.  Frees the weights that the group of `next_root` does not use.

### Backward

`ModuleParameters.backward()` starts a backward pass over the recorded order. Each region runs inside `BackwardPass.region()`, in reverse order and each once, or the region raises. The region gathers the weights as in the forward pass and prefetches the group of the region before it. Then it reduces the buckets whose gradients are complete:

*   **Gathered bucket**: The gradient buffer is allocated at the last region of the bucket in forward order, the first one of the backward pass. After the first region in forward order, the bucket reduce-scatters and adds the result to the gradient shard. It does so on every backward pass.
*   **Replicated bucket**: Autograd accumulates over all backward passes of the step, and the bucket reduces only after the final one.

A pass with `final=None`, e.g. the input pass of a split backward pass, reduces nothing. It must write no weight gradients, e.g. with `inputs=` of `torch.autograd.backward()`.

The runtime reduces only gradients that autograd accumulates in place. `backward(create_graph=True)` replaces the gradient instead, so the reduction raises.

### Finish

`wait()` raises if a trainable bucket has no reduction in a final pass, or if a gradient was replaced. Then it orders the current stream after the reductions and frees the gradient buffers and the unsharded weights. Last, it runs the post-accumulate-grad hooks of the sharded parameters. `wait()` runs once per step: a second call raises, also after a hook raised, so no hook runs twice.

A parameter that gets no gradient in the backward passes keeps a zero gradient. [Gradient Synchronization](./grad_sync.md) raises in this case instead.

Tensor hooks of a sharded parameter run each time autograd computes the local gradient of the unsharded parameter, before any reduction.

`zero_grad()` starts the next step. It clears the gradients and drops the unsharded weights. `invalidate_weights()` drops the unsharded weights before a checkpoint load.

`close()` also empties the unsharded parameters. So a graph that the caller still holds, e.g. through a logged loss, keeps no gradient buffer and no local data.

## Limits

*   The runtime needs CUDA. `bind()` and the steps must run on one stream.
*   A gathered parameter is sharded with `Shard(0)` along one dimension of `gather_dims`, with the chunk sizes of `torch.chunk`, and along no other dimension.
*   A parameter can be tied only within one registered module.
*   Every microbatch and every rank must use the same group order. Every forward pass must enter the group of every parameter.
*   Gradients must accumulate in place: `backward(create_graph=True)` raises, and a hook must not set a gradient to `None`.
*   A pass with `final=None` must write no weight gradients.
*   Weights that a region keeps or prefetches stay until `wait()`, `zero_grad()` or `invalidate_weights()`. A write to the sharded weights at another time, e.g. an optimizer step after an evaluation that left a prefetch, needs `invalidate_weights()` first.
*   Regions compute in the dtype of the sharded parameters. The runtime has no mixed precision.
*   The all-gather and the reduce-scatter of a bucket use one process group, so NCCL runs them one after the other. In the backward pass, the gather of the next group waits for the reduce-scatter of the previous one. This is a stopgap: a separate process group for the reductions must come from `DistributedContext`, which owns the process groups.

## Usage

The caller runs the regions of a module in the recorded order. Here, `model.region()` stands for the code of one region:

```python
runtime = ParameterRuntime(gather_dims={DenseDim.DP_CP_SHARD}, bucket_size_mib=32, training=True)
runtime.register(model, [model.layer])

with runtime.bind() as bound:
    parameters = bound.parameters(model)
    order = (model, model.layer, model)

    bound.zero_grad()
    regions, hidden = [], inputs
    for index, root in enumerate(order):
        region_inputs = hidden.detach().requires_grad_(index > 0)
        recorded = parameters.order
        next_root = recorded[index + 1] if recorded is not None and index + 1 < len(order) else None
        with parameters.group_weights(root, next_root):
            hidden = model.region(index, region_inputs)
        regions.append((region_inputs, hidden))
    parameters.record_order(order)

    backward, grad = parameters.backward(final=True), None
    for index in reversed(range(len(order))):
        region_inputs, outputs = regions[index]
        with backward.region(index):
            torch.autograd.backward(outputs, grad)
        grad = region_inputs.grad
    bound.wait()
    optimizer.step()
```

## API Reference

::: d9d.parallel.runtime
