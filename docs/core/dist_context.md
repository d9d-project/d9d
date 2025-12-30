---
title: Distributed Context
---

# Distributed Context

## About

The `d9d.core.dist_context` package is the **Source of Truth** for the distributed execution environment. 

In large-scale model training, ensuring that every rank agrees on the topology, global rank mapping, and communication groups is critical. This package provides the `DistributedContext` class, which serves as the central repository for this configuration. 

It is extremely important to use this context for all distributed assertions (e.g., "Am I the main process?", "Which rank is my pipeline peer?") rather than checking raw `os.environ` variables or initializing ad-hoc process groups, which can lead to silent inconsistencies.

## Comparison with Other Frameworks

The problem of managing distributed topology is solved in a different ways across different distributed training frameworks.

### Megatron-LM (`parallel_state`)

Megatron-LM manages topology via a module often called `mpu` (Model Parallel Unit) or `core.parallel_state`.

Megatron historically relies on global variables and manual rank arithmetic. To find a peer rank, developers often write code involving modulo operations (e.g., `rank % tp_size`). This is flexible but error-prone and brittle.

### HuggingFace Accelerate (`PartialState`)

Accelerate uses a class called `PartialState` to abstract the environment.

We find Accelerate's utility methods quite useful. d9d implements similar helpers, such as `wait_world()` (similar to `wait_for_everyone()`) and properties like `is_main_process` or `is_local_main_process`.

`PartialState` is primarily designed for "Flat" Data Parallelism (DDP/FSDP) and does not support complex multidimensional parallelisms natively.

`PartialState` is implemented as a Singleton. Instantiating it anywhere in the code returns the exact same global state. This makes flow of dependencies unclear and also could lead to initialization of your ProcessGroups and distributed environment in unexpected places in your code.

### TorchTitan (`ParallelDims`)

TorchTitan is the most similar framework to d9d in spirit, as both are built on top of native PyTorch 2.x `DeviceMesh` abstractions. 

However, `ParallelDims` in TorchTitan is more like a mesh factory rather than global distributed environment controller.

### d9d (`DistributedContext`)

d9d positions `DistributedContext` as the explicit controller for managing all the distributed environment.

* `DistributedContext` is a standard object that is instantiated and passed explicitly to dependent components. This ensures that the initialization of process groups happens exactly when and where the developer intends, making the initialization flow transparent.
* It replaces manual rank arithmetic with formalized and native to PyTorch `DeviceMesh` abstractions.
* Functionally, it elevates the mesh system into an active runtime controller. It bundles timeout management, context-aware logging, and node-level synchronization.


## DeviceMesh Domains

Modern architectures, especially Mixture-of-Experts (MoE), require different parallelism strategies for different parts of the model. d9d handles this by abstracting these strategies into specific **DeviceMesh Domains**.

The underlying physical GPUs are immutable, but how we view them changes depending on the layer type being processed.

### Regular Domain (`mesh_regular`)
*   **Used By**: Standard dense layers (Self-Attention, standard MLPs, RMSNorm).
*   **Dimensions**: 
    1.  `pp`: Pipeline Parallel
    2.  `dp_replicate`: Data Parallel (DistributedDataParallel style)
    3.  `dp_shard`: Data Parallel (FSDP/ZeRO style)
    4.  `cp`: Context Parallel

### Expert Domain (`mesh_ep`)
*   **Used By**: MoE layers (Expert Gate, Experts).
*   **Dimensions**:
    1.  `pp`: Pipeline Parallel (Shared with Regular Domain)
    2.  `replicate`: Replication Dimension for Experts
    3.  `ep`: Sharding Dimension for Experts

### EP Re-slicing (The Conservation of Ranks)
`mesh_ep` is not a new set of physical resources; it re-slices the ranks allocated to `mesh_regular`.

In the Expert Domain, the CP and DP dimensions (`dp_replicate`, `dp_shard`, `cp`) are collapsed and re-divided to form the Expert dimension (`ep`).

Mathematically, the total parallelism resources (within a pipeline stage) are conserved:

$$ \text{Total Ranks} = \text{DPR} \times \text{DPS} \times \text{CP} = \text{EP\_Replicate} \times \text{EP} $$

This implies that **EP takes ranks from DP and CP**.

## Usage

### Initialization
The system is usually initialized via `DeviceMeshParameters`.

```python
from d9d.core.dist_context import DeviceMeshParameters

# Define the topology
params = DeviceMeshParameters(
    pipeline_parallel=2,
    data_parallel_replicate=8,
    data_parallel_shard=1,
    context_parallel=1,
    expert_parallel=8
)

dist_ctx = params.build()
```

### Rank Utilities
Accessing rank information.

```python
if dist_ctx.is_main_process:
    print("I am Global Rank 0 (Master)")

if dist_ctx.is_local_main_process:
    print("I am Rank 0 on this specific node")

# Synchronize
dist_ctx.wait_world()
```

### Context Managers
Control execution flow across ranks.

```python
# Ensure only one process per node downloads a file
with dist_ctx.local_main_process_first():
    if dist_ctx.is_local_main_process:
        download_dataset()
    # Others wait here implicitly
# All resume together
```


::: d9d.core.dist_context
    options:
        show_root_heading: true
        show_root_full_path: true
