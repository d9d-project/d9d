# Distributed Context

## About

The `d9d.core.dist_context` package is the single source of truth for the distributed execution environment. Its `DistributedContext` class holds the topology, the rank mapping and the communication groups, so that every rank agrees on them. Use the context for all distributed questions, such as "Am I the main process?" or "Which rank is my pipeline peer?". Do not read raw `os.environ` variables or create ad-hoc process groups: this can lead to silent inconsistencies. `DistributedContext` requires CUDA GPUs. It sets the current CUDA device of each process to its local rank.

## An Explicit Object

`DistributedContext` is a plain object. You build it once from `DeviceMeshParameters`, which also creates the process groups, and pass it to every component that needs it. Besides the device meshes, it holds a logger that prefixes each line with the mesh coordinates of the rank. It also sets the timeout of all process groups and synchronizes ranks. [d9d and Other Frameworks](../concepts/comparison.md#what-a-component-can-see) compares this with the global state of other frameworks.

## DeviceMesh Domains

Different parts of a model need different parallelism strategies, for example dense layers and Mixture-of-Experts (MoE) layers. d9d describes each strategy as a **`DeviceMesh` domain**.

The physical GPUs stay the same, but each domain arranges them into a different mesh. There are domains for MoE layers, for dense layers and for the input batch. Get the `DeviceMesh` of a domain with `dist_ctx.mesh_for(domain_name)`.

!!! info "Demonstration Video"
    To understand domains better, watch our short demonstration video [on YouTube](https://www.youtube.com/watch?v=UZ2yHTGdzzU).

### Regular Domain (`regular`)

*   **Identifier**: `REGULAR_DOMAIN` or `"regular"`
*   **Purpose**: The most granular mesh, with every parallelism in its own dimension. d9d uses it for logging and seeding.
*   **Dimensions**:
    1.  `pp`: Pipeline Parallel
    2.  `dp_replicate`: Data Parallel (DDP style)
    3.  `dp_shard`: Data Parallel (FSDP style)
    4.  `cp_shard`: Context Parallel (FSDP style)
    5.  `cp_replicate`: Context Parallel (DDP style)
    6.  `tp`: Tensor Parallel

### Expert Domain (`expert`)

*   **Identifier**: `EXPERT_DOMAIN` or `"expert"`
*   **Purpose**: The mesh for MoE layers. Shard sparse expert layers across the `ep_shard` dimension and replicate them across the `ep_replicate` dimension.
*   **Dimensions**:
    1.  `pp`: Pipeline Parallel
    2.  `ep_replicate`: Combined replication dimension (`(DP * CP) // EP`)
    3.  `ep_shard`: Expert Parallel

### Dense Domain (`dense`)

*   **Identifier**: `DENSE_DOMAIN` or `"dense"`
*   **Purpose**: The mesh for dense layers.
*   **Dimensions**:
    1.  `pp`: Pipeline Parallel
    2.  `dp_replicate`: Data Parallel replication in HSDP
    3.  `dp_cp_shard`: Data Parallel and Context Parallel merged, for sharding in HSDP
    4.  `cp_replicate`: Context Parallel replication
    5.  `tp`: Tensor Parallel

### Batch Domain (`batch`)

*   **Identifier**: `BATCH_DOMAIN` or `"batch"`
*   **Purpose**: The mesh for distributing the input batch and sharding the data loader.
*   **Dimensions**:
    1.  `pp`: Pipeline Parallel
    2.  `dp`: Data Parallel (`dp_replicate * dp_shard`)
    3.  `cp`: Context Parallel (`cp_replicate * cp_shard`)
    4.  `tp`: Tensor Parallel

### Flat Domain (`flat`)

*   **Identifier**: `FLAT_DOMAIN` or `"flat"`
*   **Purpose**: A mesh with a single dimension that holds all processes.
*   **Dimensions**:
    1.  `world`: World size

## Usage

### Initialization

Build the context from `DeviceMeshParameters`.

```python
from d9d.core.dist_context import DeviceMeshParameters

# Define the topology.
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

```python
if dist_ctx.is_main_process:
    print("I am global rank 0")

if dist_ctx.is_local_main_process:
    print("I am rank 0 on this node")

# Synchronize.
dist_ctx.wait_world()
```

### Context Managers

These context managers control the order in which ranks run a block.

```python
# Only one process per node downloads the file.
with dist_ctx.local_main_process_first():
    if dist_ctx.is_local_main_process:
        download_dataset()
    # Other ranks enter the block after the download.
# All ranks continue together.
```

## API Reference

::: d9d.core.dist_context
