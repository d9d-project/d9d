---
title: Gradient Norm & Clipping
---

# Gradient Norm & Clipping

## About

!!! warning "Warning:" 
    If you are utilizing the standard `d9d` training infrastructure, you **do not** need to call these functions manually. The framework automatically handles gradient clipping. This package is primarily intended for users extending the internals of `d9d`.

The `d9d.internals.grad_norm` package handles the calculation and clipping of gradient norms in complex distributed environments.

Standard PyTorch `clip_grad_norm_` functions are not fully aware of heterogeneous ND-Parallelism strategies (mixing Pipeline, Data, Tensor, and Context Parallelism). This package ensures that the global norm is correctly calculated across all parallel dimensions and that `DTensor` sharding is handled without unnecessary full-tensor materialization.

## Concepts

### Distributed Heterogeneity

Some parameters might be `Shard`ed across a TP/FSDP mesh, while others are `Replicate`d. Also model may be pipelined.

To handle this, we decompose the problem:

1.  **Local Norm**: Calculate the norm of the tensor shards actually present in GPU memory (using `to_local()`).
2.  **Horizontal Reduction**: Perform `all_reduce` strictly on the meshes where parameters are sharded. This ensures that sharded parameters contribute correctly to the global norm, while replicated parameters do not trigger double-counting or unnecessary communication for norm calculation.
3.  **Pipeline Reduction**: Finally, norms are summed across the Pipeline Parallel mesh, as different stages hold completely different parameters.

### Grouping & Overlap

To optimize performance, `group_parameters_for_norm` groups parameters into `GradNormGroup` buckets. This grouping is based on:

1.  **Sharding Strategy**: Parameters sharded on the same mesh are grouped together so their norms can be reduced in a single collective operation.
2.  **Device & DType**: Ensures compatibility for local math operations.

The system attempts to overlap communication with computation. Groups containing sharded tensors are prioritized so their `all_reduce` operations can run asynchronously while local norms for other groups are being computed.


## Mathematical Correctness

The goal of distributed gradient clipping is to calculate the **Global Norm** ($\|\mathbf{g}\|$) of a **single model instance**, regardless of how that model is physically fragmented across GPUs.

Let the total set of model parameters $\mathcal{P}$ be divided into disjoint subsets based on parallelism strategy:
1.  $\mathcal{P}_{pp}$: Sets of parameters residing on different Pipeline stages.
2.  $\mathcal{P}_{sharded}$: Parameters split across a TP/EP/FSDP group.
3.  $\mathcal{P}_{repl}$: Parameters replicated across other groups.

The definition of the global $L_2$ norm is:

$$ \|\mathbf{g}\|_2 = \sqrt{ \sum_{p \in \mathcal{P}} \|g_p\|^2 } $$

We prove that our strategy of separating aggregation logic based on placement prevents double-counting.

### Proof for Sharded Parameters (TP/EP/FSDP)
For a parameter $w \in \mathcal{P}_{sharded}$, the logical gradient tensor $G$ is split into physical shards $G_1, G_2, \dots, G_k$ across $k$ devices. By the definition of the Frobenius norm:

$$ \|G\|^2 = \sum_{rank=1}^{k} \|G_{rank}\|^2 $$

**Strategy:** We calculate local norms and apply `all_reduce(op=SUM)`.

### Proof for Replicated Parameters (DP)
For a parameter $w \in \mathcal{P}_{repl}$, the logical gradient tensor $G$ is identical on all $k$ devices (assuming DP synchronization has occurred).
$$ G_{rank_1} = G_{rank_2} = \dots = G $$

If we were to sum these (as we did for TP), we would obtain:
$$ \sum_{rank=1}^{k} \|G_{rank}\|^2 = k \cdot \|G\|^2 \quad (\text{Incorrect: Double Counting}) $$

**Strategy:** We group these parameters separately and do not communicate.

### Proof for Pipeline Parallelism (PP)
Pipeline stages hold disjoint sets of parameters. The total norm is simply the sum of the norms of the stages.

$$ \|\mathbf{g}\|^2 = \|\mathbf{g}_{stage_1}\|^2 + \|\mathbf{g}_{stage_2}\|^2 + \dots $$

**Strategy:** We apply `all_reduce(op=SUM)` across the PP mesh.

### Result
The final formula utilized by `d9d` ensures $1:1$ correspondence with a single-device baseline:

$$ \|\mathbf{g}\|_{global} = \sqrt{ \underbrace{\sum_{pp} \left( \underbrace{\sum_{tp} \|g_{sharded}\|^2}_{\text{Sum Unique Shards}} + \underbrace{\|g_{replicated}\|^2}_{\text{Do Not Duplicate}} \right)}_{\text{Sum Disjoint Layers}} } $$


::: d9d.internals.grad_norm
    options:
        show_root_heading: true
        show_root_full_path: true
