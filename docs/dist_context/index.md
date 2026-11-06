# Distributed Context

## About

The `d9d.dist_context` package holds the distributed state of a process. Its `DistributedContext` knows the rank, the GPU of the rank, the process groups and the device meshes of the domains. Every component that needs these facts gets the context through its constructor. Do not read raw `os.environ` variables or create ad-hoc process groups: then the components can disagree about the topology. `DistributedContext` requires CUDA GPUs.

## Creating and Closing the Context

`init_dist_context()` creates the context from `DeviceMeshParameters`, once per process. In a distributed job, every rank calls it with the same arguments. It takes the world size and the rank from the default process group if one exists, and otherwise from `WORLD_SIZE` and `RANK` of the launcher. Without a launcher, the job has one rank and runs with `python`. The product of the four parallelism degrees must equal the world size.

`init_dist_context()` is a context manager, and the context is valid only inside its `with` block. On exit, it destroys the groups that it created, including the default process group. If the block raises, the groups stay, so the job fails instead of hanging.

In a distributed job, the block also replaces the `DTensor` RNG tracker of the process. The default tracker of PyTorch broadcasts the generator state of rank 0, which would overwrite the seeds of the other pipeline stages. The previous tracker comes back on exit, also after an error.

A job takes the context and never closes it. `TrainingConfigurator` and `InferenceConfigurator` take it as `dist_context`, so one context can serve several jobs of a process, e.g. training and then evaluation.

## GPU Assignment

The degrees describe the ranks, not the GPUs. The same configuration runs on 8 GPUs and on 2, so the context takes the GPU of each rank from the launch:

*   **Local rank**: It comes from `LOCAL_RANK`, which `torchrun` sets. It is the index of the rank on its node, whatever its GPU.
*   **GPU**: The node runs `LOCAL_WORLD_SIZE` ranks on `torch.cuda.device_count()` visible GPUs. If the ranks do not outnumber the GPUs, local rank `i` runs on GPU `i`. Otherwise each GPU runs an equal block of consecutive local ranks. E.g. with 8 ranks on 2 GPUs, local ranks 4 to 7 run on GPU 1. The context raises if the counts do not divide.

Several ranks on one GPU let you check a parallel layout on fewer GPUs than it targets. They need `NCCL_MULTI_RANK_GPU_ENABLE=1` from the launcher. [CONTRIBUTING.md](https://github.com/d9d-project/d9d/blob/main/CONTRIBUTING.md#distributed-tests-on-fewer-gpus) shows the setup for the tests of d9d.

## Parallelism Degrees

The work of one step splits in four ways. `DeviceMeshParameters` holds one degree for each:

*   **`pipeline_parallel`**: It splits the layers into stages.
*   **`data_parallel`**: It splits the samples.
*   **`context_parallel`**: It splits one sample into parts, e.g. a text into chunks of the sequence, a graph into subgraphs or an image into tiles.
*   **`tensor_parallel`**: It splits the matrix multiplications inside one sample.

Other kinds of parallelism split the same ranks differently, so they do not add to the world size. Each built-in domain that does so has its own section:

*   **`dense`**: The sizes of the shard groups of the dense layers, `data_parallel_shard` and `context_parallel_shard`. They default to 1, so the dense layers do not shard, as in DDP. The number of replicas follows from the degree, so a shard size that fits the memory of a GPU stays when the job grows.
*   **`expert`**: The degree of expert parallelism, `expert_parallel`. Without this section, the job builds no expert domain.

The shard sizes must divide `data_parallel` and `context_parallel`, and `expert_parallel` must divide `data_parallel * context_parallel`.

## Domains

Different parts of a model need different parallelism, e.g. dense layers and Mixture-of-Experts (MoE) layers. d9d describes each arrangement of the ranks as a **domain**. All domains hold the same ranks, and each arranges them into its own device mesh. The context keeps each mesh under its domain object: `dist_context.mesh_for(DENSE_DOMAIN)`.

| Domain | Dimensions | Purpose |
|:-------|:-----------|:--------|
| `BASE_DOMAIN` | `BaseDim`: `PP`, `DP`, `CP`, `TP` | Pipeline stages, data and the coordinates of a rank |
| `DENSE_DOMAIN` | `DenseDim`: `PP`, `DP_REPLICATE`, `DP_CP_SHARD`, `CP_REPLICATE`, `TP` | Dense layers |
| `EXPERT_DOMAIN` | `ExpertDim`: `PP`, `EP_REPLICATE`, `EP_SHARD`, `TP` | Expert layers |
| `METRIC_DOMAIN` | `MetricDim`: `PP_DP_CP`, `TP` | Sums of the metrics and the loss weights |

*   **Base domain**: Its dimensions are the four degrees. Their order defines how the ranks lie in the world: the ranks of one tensor-parallel group are neighbors.
*   **Dense domain**: It splits `dp` into `DP_REPLICATE` and a shard, and `cp` into a shard and `CP_REPLICATE`, as the `dense` section says. `DP_CP_SHARD` merges the two shards.
*   **Expert domain**: The context builds it only if the `expert` section is set. Otherwise `mesh_for(EXPERT_DOMAIN)` raises and asks to set the section. `EP_REPLICATE` and `EP_SHARD` split the ranks of `dp * cp`. So an expert-parallel group never holds two ranks of one tensor-parallel group, which hold the same tokens.
*   **Metric domain**: `PP_DP_CP` holds all ranks with their own tokens or pipeline stage. The gradient manager and the metrics sum over it, so the ranks of one tensor-parallel group count once.

Each domain names its dimensions with a `StrEnum`, e.g. `DenseDim.DP_REPLICATE`, so the type checker catches a dimension of another domain.

## Coordinates and Communication

`DistributedContext` answers four kinds of questions. Only communication needs process groups:

| Kind | Members | In a job without parallelism |
|:-----|:--------|:-----------------------------|
| Rank | `rank`, `world_size`, `is_main_process`, `local_rank`, `is_local_main_process`, `current_device` | Rank 0 of 1 |
| Coordinates | `coordinate()`, `dim_size()` | Coordinate 0, size 1 |
| Communication | `is_distributed`, `mesh_for()`, `wait_world()`, `main_process_first()`, `local_main_process_first()`, `set_timeout()` | `is_distributed` is `False` and `mesh_for()` raises. `set_timeout()` does nothing. The others only wait for the GPU. |

Code that communicates checks `is_distributed` before it calls `mesh_for()`.

## Extra Domains

A new kind of parallelism adds a domain that splits the ranks differently. For example, an image model can split the `cp` ranks into height and width. Subclass `DeviceMeshDomain` and return the dimensions of the mesh from `mesh_dims()`, outermost first. The context lays out the ranks along them row-major, as `init_device_mesh()` does. So dimensions that keep the order of the base dimensions group the ranks as they do. A domain takes its own degrees in its constructor.

Pass the domain in `extra_domains` of `init_dist_context()`. The sizes of its dimensions must be positive and multiply to the world size, and the names of its dimensions must be unique. The names of all domains must be unique too. `mesh_for()` looks the mesh up by the domain object, so a domain object that was not passed raises, even if its name matches.

## Usage

### Configuration

A job with pipeline parallelism, HSDP inside each node and expert parallelism:

```yaml
mesh:
  pipeline_parallel: 2
  data_parallel: 16
  dense:
    data_parallel_shard: 8
  expert:
    expert_parallel: 8
```

### Creating the Context

```python
from d9d.dist_context import DeviceMeshParameters, init_dist_context
from d9d.loop.run import TrainingConfigurator

params = DeviceMeshParameters(data_parallel=8)

with init_dist_context(params) as dist_context:
    trainer = TrainingConfigurator(
        dist_context=dist_context,
        # The parameters and the providers of the job.
    ).configure()
    trainer.train()
```

### Coordinates and Meshes

```python
from d9d.dist_context import BASE_DOMAIN, DENSE_DOMAIN, BaseDim, DenseDim, DistributedContext

dist_context: DistributedContext = ...

# Works with and without parallelism.
dp_rank = dist_context.coordinate(BASE_DOMAIN, BaseDim.DP)
dp_size = dist_context.dim_size(BASE_DOMAIN, BaseDim.DP)

if dist_context.is_distributed:
    shard_mesh = dist_context.mesh_for(DENSE_DOMAIN)[DenseDim.DP_CP_SHARD]
```

### An Extra Domain

```python
from enum import StrEnum

from d9d.dist_context import DeviceMeshDomain, DeviceMeshParameters, MeshDim, init_dist_context


class DataContextParallelDim(StrEnum):
    PP = "pp"
    DP_CP = "dp_cp"
    TP = "tp"


class DataContextParallelDomain(DeviceMeshDomain[DataContextParallelDim]):
    @property
    def name(self) -> str:
        return "dp_cp"

    def mesh_dims(self, params: DeviceMeshParameters) -> tuple[MeshDim[DataContextParallelDim], ...]:
        return (
            MeshDim(DataContextParallelDim.PP, params.pipeline_parallel),
            MeshDim(DataContextParallelDim.DP_CP, params.data_parallel * params.context_parallel),
            MeshDim(DataContextParallelDim.TP, params.tensor_parallel),
        )


DP_CP_DOMAIN = DataContextParallelDomain()

with init_dist_context(params, extra_domains=(DP_CP_DOMAIN,)) as dist_context:
    shard_index = dist_context.coordinate(DP_CP_DOMAIN, DataContextParallelDim.DP_CP)
```

### Ordering the Ranks

```python
# Only one process per node downloads the file.
with dist_context.local_main_process_first():
    if dist_context.is_local_main_process:
        download_dataset()
    # Other ranks enter the block after the download.
# All ranks continue together.

# Wait for all ranks.
dist_context.wait_world()
```

## API Reference

::: d9d.dist_context
