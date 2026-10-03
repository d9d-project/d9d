# Gradient Norm and Clipping

## About

The `d9d.internals.grad_norm` package computes and clips gradient norms in distributed jobs. The standard PyTorch `clip_grad_norm_` does not know about ND parallelism, which mixes pipeline, data, tensor and context parallelism. This package computes the global norm correctly across all parallel dimensions. It handles `DTensor` sharding without materializing full tensors. Tensors sharded on more than one mesh dimension are not supported.

!!! warning "Internal API"
    If you use the standard d9d training loop, you do not need to call this package. d9d clips the gradients itself. This page is for users who extend the internals of d9d.

## Distributed Heterogeneity

Some parameters can be `Shard`ed across a TP or FSDP mesh, while others are `Replicate`d. The model can also be pipelined.

So the computation has three steps:

1.  **Local norm**: Compute the norm of the tensor shards present in GPU memory (with `to_local()`).
2.  **Horizontal reduction**: Run `all_reduce` only on the meshes where parameters are sharded. Sharded parameters then contribute correctly to the global norm. Replicated parameters are not counted twice and need no communication.
3.  **Pipeline reduction**: Sum the norms across the pipeline parallel mesh, because different stages hold different parameters.

For the max norm (`inf`), both reductions take the maximum instead of the sum.

## Grouping and Overlap

`group_parameters_for_norm` groups parameters into `GradNormGroup` buckets by:

1.  **Sharding**: Parameters sharded on the same mesh share one collective for their norm.
2.  **Device and dtype**: Parameters in a group must be compatible for local math.

Groups of sharded tensors come first. Their `all_reduce` runs asynchronously while the local norms of the other groups are computed.

## Mathematical Correctness

Distributed gradient clipping must compute the **global norm** ($\|\mathbf{g}\|$) of a **single model instance**, however the model is split across GPUs.

Split the set of model parameters $\mathcal{P}$ into disjoint subsets by parallelism strategy:

1.  $\mathcal{P}_{pp}$: the sets of parameters on different pipeline stages.
2.  $\mathcal{P}_{sharded}$: parameters split across a TP, EP or FSDP group.
3.  $\mathcal{P}_{repl}$: parameters replicated across other groups.

The global $L_2$ norm is defined as:

$$ \|\mathbf{g}\|_2 = \sqrt{ \sum_{p \in \mathcal{P}} \|g_p\|^2 } $$

The proofs below show that treating each placement separately prevents double counting.

### Proof for Sharded Parameters (TP/EP/FSDP)

For a parameter $w \in \mathcal{P}_{sharded}$, the logical gradient tensor $G$ is split into physical shards $G_1, G_2, \dots, G_k$ across $k$ devices. By the definition of the Frobenius norm:

$$
\|G\|^2 = \sum_{rank=1}^{k} \|G_{rank}\|^2
$$

**Strategy:** compute the local norms and apply `all_reduce(op=SUM)`.

### Proof for Replicated Parameters (DP)

For a parameter $w \in \mathcal{P}_{repl}$, the logical gradient tensor $G$ is the same on all $k$ devices, once DP synchronization has happened.

$$
G_{rank_1} = G_{rank_2} = \dots = G
$$

Summing them like the sharded case would give:

$$
\sum_{rank=1}^{k} \|G_{rank}\|^2 = k \cdot \|G\|^2 \quad (\text{Incorrect: Double Counting})
$$

**Strategy:** group these parameters separately and do not communicate.

### Proof for Pipeline Parallelism (PP)

Pipeline stages hold disjoint sets of parameters. The total norm is the sum of the norms of the stages.

$$
\|\mathbf{g}\|^2 = \|\mathbf{g}_{stage_1}\|^2 + \|\mathbf{g}_{stage_2}\|^2 + \dots
$$

**Strategy:** apply `all_reduce(op=SUM)` across the PP mesh.

### Result

d9d uses the formula below. It gives the same norm as a single-device baseline:

$$
\|\mathbf{g}\|_{global} = \sqrt{ \underbrace{\sum_{pp} \left( \underbrace{\sum_{tp} \|g_{sharded}\|^2}_{\text{Sum Unique Shards}} + \underbrace{\|g_{replicated}\|^2}_{\text{Do Not Duplicate}} \right)}_{\text{Sum Disjoint Layers}} }
$$

## API Reference

::: d9d.internals.grad_norm
