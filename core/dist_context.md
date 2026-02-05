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

Modern architectures require different parallelism strategies for different parts of the model (e.g., standard dense layers vs. Mixture-of-Experts layers). d9d handles this by abstracting these strategies into specific **DeviceMesh Domains**.

The underlying physical GPUs are immutable, but how we view them changes depending on what we are working with (distributing MoE layers, Dense layers, distributing input batch). `DeviceMesh` object for specific domain is retrieved via `dist_ctx.mesh_for(domain_name)`.

!!! info "Demonstration Video: "
    For better understanding domains, we have prepared a quick demonstration video [on YouTube](https://www.youtube.com/watch?v=UZ2yHTGdzzU).

### Regular Domain (`regular`)

*   **Identifier**: `REGULAR_DOMAIN` or `"regular"`
*   **Purpose**: The most granular mesh view for fully decomposed parallelism. Used for setting up logging and seeding.
*   **Dimensions**: 
    1.  `pp`: Pipeline Parallel
    2.  `dp_replicate`: Data Parallel (DDP style)
    3.  `dp_shard`: Data Parallel (FSDP style)
    4.  `cp_shard`: Context Parallel (FSDP style)
    5.  `cp_replicate`: Context Parallel (DDP style)
    6.  `tp`: Tensor Parallelism

### Expert Domain (`expert`)

*   **Identifier**: `EXPERT_DOMAIN` or `"expert"`
*   **Purpose**: Mesh view optimized for distributing MoE (Mixture of Experts) layers. It is intended that sparse expert layers should be sharded across `ep_shard` dimension and replicated across `ep_replicate` dimension.
*   **Dimensions**:
    1.  `pp`: Pipeline Parallel
    2.  `ep_replicate`: Combined Replication Dimension (`(DP * CP) // EP`)
    3.  `ep_shard`: Expert Parallel Dimension

### Dense Domain (`dense`)

*   **Identifier**: `DENSE_DOMAIN` or `"dense"`
*   **Purpose**: Mesh view for distributing dense layers.
*   **Dimensions**:
    1.  `pp`: Pipeline Parallel
    2.  `dp_replicate`: Data Parallel for replication using HSDP
    3.  `dp_cp_shard`: Merged Data and Context Parallel dimension for sharding using HSDP
    4.  `cp_replicate`: Context Parallel for replication
    5.  `tp`: Tensor Parallel

### Batch Domain (`batch`)

*   **Identifier**: `BATCH_DOMAIN` or `"batch"`
*   **Purpose**: Mesh view for distributing batch tensor and setting up DataLoader sharding.
*   **Dimensions**:
    1.  `pp`: Pipeline Parallel
    2.  `dp`: Data Parallel
    3.  `cp`: Context Parallel
    4.  `tp`: Tensor Parallel


### Flat Domain (`flat`)

*   **Identifier**: `FLAT_DOMAIN` or `"flat"`
*   **Purpose**: Mesh view with a single dimension.
*   **Dimensions**:
    1.  `world`: World Size

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
    context_parallel_replicate=1,
    context_parallel_shard=1,
    expert_parallel=8,
    tensor_parallel=1
)

dist_ctx = params.build()
```

### Accessing DeviceMesh Domains

```python
from torch.distributed import DeviceMesh
from d9d.core.dist_context import DistributedContext, DENSE_DOMAIN

dist_ctx: DistributedContext = ...

mesh_dense: DeviceMesh = dist_ctx.mesh_for(DENSE_DOMAIN)
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
