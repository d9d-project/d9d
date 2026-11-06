---
DEP: 0012
Title: Rework of DistributedContext
Author: Maksim Afanasev @mrapplexz
Type: Feature
---

# Rework of DistributedContext

## Abstract

`DistributedContext` binds each rank to its own GPU and builds a fixed set of domains, each with its own process groups. So a job cannot run several ranks on one GPU or add a domain.

This DEP creates the context in a function, whose caller owns and closes it. The context can assign several ranks to one GPU. One base mesh holds the four degrees of parallelism, and every other domain declares how it splits the same ranks differently. The context creates the groups of each distinct partition once.

## Motivation

Issue [#107](https://github.com/d9d-project/d9d/issues/107) tracks this rework.

### The context binds each rank to its own GPU

*   **One rank per GPU**: The context and `init_device_mesh()` call `torch.cuda.set_device(LOCAL_RANK)`. So the 8-rank distributed tests fail on 2 GPUs, although NCCL 2.30 allows it with the experimental [`NCCL_MULTI_RANK_GPU_ENABLE`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-multi-rank-gpu-enable).
*   **The workaround**: A wrapper that sets `CUDA_VISIBLE_DEVICES` per rank and `LOCAL_RANK=0` makes every rank a local main process.
*   **Node properties**: `num_nodes` and `node_rank` divide by `torch.cuda.device_count()`, so 8 ranks on 2 GPUs give `num_nodes` of 4. No code in d9d reads them or `master_addr`.

### The domains are closed

*   **Fixed list**: A job can add a domain only by mutating the module-level list `ALL_DOMAIN_PROVIDERS`, which d9d neither exports nor documents.
*   **String names**: `mesh_for()` and 28 mesh slices in the sources and examples take string literals, so the type checker catches no typo.
*   **Duplicate groups**: Each domain calls `init_device_mesh()` and creates its own groups. With `pp`, `dp_replicate`, `dp_shard` and `ep` of 2, the built-in domains create 18 groups for 6 distinct partitions of the ranks.
*   **Branches**: A job without parallelism has no meshes. So 10 places branch only to learn where the rank is, e.g. `mesh_for(BATCH_DOMAIN)["dp"].get_local_rank() if is_distributed else 0`.

### The configuration mixes the split of the work with the storage of parameters

*   **Mixed degrees**: The configuration gives data and context parallelism only as replicate and shard degrees, which say how FSDP stores the parameters. So the regular domain has six dimensions, although the work splits in four ways.
*   **Implicit expert mesh**: `expert_parallel` defaults to 1, so every job builds the expert mesh, and a model with experts silently runs without expert parallelism.
*   **Configuration through the context**: Components read `dist_context.mesh_params` to learn whether they can communicate. `DeviceMeshParameters` also carries 7 `has_*` properties that only compare a degree with 1.

### The context does not manage its resources

*   **Logger**: The constructor replaces the handlers of the `d9d` logger. A component that only logs must take the whole context for `dist_context.logger`.
*   **Closing**: d9d never calls `torch.distributed.destroy_process_group()`, so every distributed job ends with a warning of PyTorch.
*   **One context per job**: Each configurator builds its own context. A process that trains and then evaluates builds two contexts with two sets of groups.

### Bugs

*   **World size**: If all degrees are 1, the context takes the single-process path even under `torchrun`, and each rank trains its own copy ([#78](https://github.com/d9d-project/d9d/issues/78)). This DEP takes over the check of [#95](https://github.com/d9d-project/d9d/pull/95).
*   **Expert mesh**: The validation checks that `ep` divides `dp * cp * tp`, but the expert domain leaves out `tp`. So with `tp > 1`, the expert mesh holds only the first `world / tp` ranks.
*   **Tensor parallelism**: `GradientManager` sums the loss weights over all ranks, but the ranks of one tensor-parallel group share their tokens. So the gradients and their norm come out `tp` times too small, and the metrics count each sample `tp` times. A job with its own tensor parallelism gets this silently.

## Design Proposal

### Building the Context

This DEP introduces `init_dist_context()`. It replaces `DeviceMeshParameters.build()` and the public constructor of `DistributedContext`, because a configuration must not build runtime objects. Like `init_device_mesh()` of PyTorch, it initializes the distributed state of the process.

```python
def init_dist_context(
    params: DeviceMeshParameters,
    extra_domains: Sequence[DeviceMeshDomain] = (),
) -> DistributedContext:
    """Builds the distributed context of this process and sets its current GPU.

    In a distributed job, every rank must call it with the same ``params`` and the same ``extra_domains``.

    Args:
        params: The parallelism degrees and the sections of the built-in domains.
        extra_domains: The domains to build after the built-in domains.

    Returns:
        The distributed context. The caller must close it.

    Raises:
        ValueError: If the degrees do not match the world size or the GPUs, if a domain is invalid, or if the
            ranks pass different domains.
    """
```

`TrainingConfigurator` and `InferenceConfigurator` take a `dist_context` instead of `mesh`, so one context serves all jobs of a process, e.g. training and then evaluation. Each job keeps its own `TimeoutManager`, which sets `init_timeout` when the job starts.

`init_dist_context()` runs these steps in this order:

1.  **World**: It takes the world size and the global rank from the default process group if one exists. Otherwise it takes them from `WORLD_SIZE` and `RANK` of the launcher. Without `WORLD_SIZE`, the world has one rank, so a single-process job runs with `python`. It raises if the product of the degrees differs from the world size.
2.  **GPU**: It assigns the rank to a GPU and makes it the current GPU (see GPU Assignment).
3.  **Default process group**: In a distributed job, it creates the default process group, unless one exists.
4.  **Consistency**: It gathers the names and dimensions of all domains from all ranks over the default group. It raises on all ranks if they differ. Group creation is collective, so a mismatch would hang otherwise.
5.  **Validation**: It checks every domain (see Domains), also in a job without parallelism. All ranks check the same data, so they all raise together.
6.  **Groups and meshes**: It creates the process groups and builds the meshes (see Domains).

### GPU Assignment

The degrees describe the ranks, not the GPUs they run on. The same configuration must run on 8 GPUs and on 2, so the context takes the GPU assignment from the launch.

*   **Ranks**: The local rank comes from `torch.distributed.get_node_local_rank()`. It stays the index of the rank on its node, whatever its GPU, so `is_local_main_process` holds on one rank per node. With `torchrun --virtual-local-rank`, every rank has local rank 0 and its own GPU.
*   **GPU**: The node runs `LOCAL_WORLD_SIZE` ranks on `torch.cuda.device_count()` visible GPUs. If the ranks do not outnumber the GPUs, local rank `i` runs on GPU `i`. Otherwise each GPU runs an equal block of consecutive local ranks, and the context raises if the counts do not divide. Without `LOCAL_WORLD_SIZE`, local rank `i` runs on GPU `i`.
*   **Current GPU**: The context sets the GPU before it creates any process group. `current_device` returns the GPU with its index, e.g. `cuda:3`.
*   **Consent**: Several ranks on one GPU need `NCCL_MULTI_RANK_GPU_ENABLE=1` from the launcher. d9d does not check it, because NCCL names the variable in its error and can also read it from a config file.

### Parameters

The work of one step splits in four ways:

*   **Pipeline parallelism**: It splits the layers into stages.
*   **Data parallelism**: It splits the samples.
*   **Context parallelism**: It splits one sample into parts, e.g. a text into chunks of the sequence, a graph into subgraphs or an image into tiles.
*   **Tensor parallelism**: It splits the matrix multiplications inside one sample.

Every other kind of parallelism splits the same ranks differently. For example, FSDP splits the ranks of `dp * cp` into replicas and shards, and expert parallelism splits them into groups of experts. So `DeviceMeshParameters` holds the four degrees and one section for each built-in domain that splits them differently:

```python
class DenseDomainConfig(BaseModel):
    """Configuration for the dense domain, which shards the parameters of dense layers.

    Attributes:
        data_parallel_shard: The number of data-parallel ranks that shard a parameter.
        context_parallel_shard: The number of context-parallel ranks that shard a parameter.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_parallel_shard: PositiveInt = 1
    context_parallel_shard: PositiveInt = 1

class ExpertDomainConfig(BaseModel):
    """Configuration for the expert domain.

    Attributes:
        expert_parallel: The number of ranks that split the experts of a layer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    expert_parallel: PositiveInt

class DeviceMeshParameters(BaseModel):
    """Configuration for the parallelism degrees of a job.

    Attributes:
        pipeline_parallel: The degree of pipeline parallelism.
        data_parallel: The degree of data parallelism.
        context_parallel: The degree of context parallelism.
        tensor_parallel: The degree of tensor parallelism.
        dense: The configuration of the dense domain.
        expert: The configuration of the expert domain. ``None`` means that the job builds no expert domain.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pipeline_parallel: PositiveInt = 1
    data_parallel: PositiveInt = 1
    context_parallel: PositiveInt = 1
    tensor_parallel: PositiveInt = 1
    dense: DenseDomainConfig = Field(default_factory=DenseDomainConfig)
    expert: ExpertDomainConfig | None = None
```

*   **Shards, not replicas**: The dense section gives the size of a shard group, and the number of replicas follows from the degree. The shard group depends on the memory of a GPU and the topology of a node, not on the scale of the job.
*   **Defaults**: Without the dense section, the dense layers do not shard, as in DDP. Sharding saves memory at the cost of communication, so the user enables it, like today.
*   **Validation**: Pydantic checks that the shard sizes divide `dp` and `cp`, and that `expert_parallel` divides `dp * cp`, when it reads the configuration. Unknown keys fail, so an old configuration does not fall back to DDP silently.
*   **No runtime state**: `build()` moves to `init_dist_context()`, `is_distributed` moves to the context, and the `has_*` properties go away. The context no longer exposes `mesh_params`.

### Domains

This DEP makes `DeviceMeshDomain` public in `d9d.dist_context`. A domain declares the dimensions of its mesh and does not build it. The context sees the dimensions of all domains at once, so it decides alone how to create the process groups. This DEP also introduces `MeshDim`.

```python
DimT = TypeVar("DimT", bound=StrEnum)

@dataclass(frozen=True)
class MeshDim(Generic[DimT]):
    """One dimension of a device mesh.

    Attributes:
        name: The name of the dimension.
        size: The number of ranks along the dimension.
    """

    name: DimT
    size: int

class DeviceMeshDomain(abc.ABC, Generic[DimT]):
    """Abstract base class for a domain: one arrangement of all ranks into a device mesh."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of this domain, unique within a context."""

    @abc.abstractmethod
    def mesh_dims(self, params: DeviceMeshParameters) -> tuple[MeshDim[DimT], ...]:
        """Returns the dimensions of the mesh of this domain, outermost first.

        The context lays out the ranks along them row-major, as ``init_device_mesh()`` does. So a dimension that
        keeps the order of the dimensions of ``BASE_DOMAIN`` groups the ranks as they do.

        Args:
            params: The parallelism degrees and the sections of the built-in domains.

        Returns:
            The dimensions of the mesh.
        """
```

The domain constants become the domain objects, e.g. `DENSE_DOMAIN = _DenseDomain()`. The built-in domain classes stay private. `mesh_for(domain: DeviceMeshDomain)` replaces `mesh_for(domain: str)`.

The built-in domains:

| Domain | Dimensions | Purpose |
|:-------|:-----------|:--------|
| `BASE_DOMAIN` | `BaseDim`: `PP`, `DP`, `CP`, `TP` | Pipeline stages, data and the coordinates of a rank |
| `DENSE_DOMAIN` | `DenseDim`: `PP`, `DP_REPLICATE`, `DP_CP_SHARD`, `CP_REPLICATE`, `TP` | Dense layers |
| `EXPERT_DOMAIN` | `ExpertDim`: `PP`, `EP_REPLICATE`, `EP_SHARD`, `TP` | Expert layers |
| `METRIC_DOMAIN` | `MetricDim`: `PP_DP_CP`, `TP` | Sums of the metrics and the loss weights |

*   **Base domain**: Its dimensions are the four degrees, and their order defines how the ranks lie in the world. It replaces `REGULAR_DOMAIN` and `BATCH_DOMAIN`.
*   **Dense domain**: It splits `dp` into `DP_REPLICATE` and a shard, and `cp` into a shard and `CP_REPLICATE`, as the dense section says. `DP_CP_SHARD` merges the two shards.
*   **Expert domain**: The context builds it only if the expert section is set. Otherwise `mesh_for(EXPERT_DOMAIN)` raises and asks to set it, so a distributed job with experts fails in `configure()`. A job without parallelism never asks for the mesh, so it runs without the section. `EP_REPLICATE` and `EP_SHARD` split the ranks of `dp * cp`. So expert parallelism never groups the ranks of one tensor-parallel group, which hold the same tokens.
*   **Metric domain**: It is new. `PP_DP_CP` holds all ranks with their own tokens or stage. `GradientManager` and the metrics reduce over it, so tensor-parallel ranks count once.
*   **No flat domain**: `FLAT_DOMAIN` goes away. The loss and the metrics move to `METRIC_DOMAIN`, and a group of all ranks is the default process group. The `DTensor` RNG tracker uses the base mesh.
*   **New parallelism**: A new kind of parallelism adds a domain that splits the base ranks differently. For example, an image model splits the `cp` ranks into height and width. The domain takes its own degrees in its constructor.

A domain is generic over a `StrEnum` of the names of its dimensions, e.g. `DeviceMeshDomain[BaseDim]`. Its `mesh_dims()`, `coordinate()`, `dim_size()` and all code that slices its mesh use the members. So the type checker catches a typo and a dimension of another domain, e.g. `coordinate(DENSE_DOMAIN, BaseDim.DP)`. A member is a `str`, so `DeviceMesh` takes it as a dimension name.

*   **Validation**: The sizes of the dimensions of a domain must be positive and multiply to the world size, and the names of its dimensions must be unique. The names of the domains must be unique too.
*   **Groups**: Each dimension partitions the ranks into groups along it. The context creates the groups of each distinct partition once, across all domains of the context. So two domains with `pp` share its groups, and a dimension of all ranks uses the default group.
*   **Size 1**: All dimensions of size 1 have the same partition, so they share one set of groups. NCCL creates the communicator of a group at its first collective, so a group that never communicates gets none. This holds unless the caller created the default group with `device_id`: then PyTorch splits every subgroup from it at creation. The `fake` backend of PyTorch would avoid even that, but it registers only through `torch.testing._internal`.
*   **Meshes**: The context builds each mesh from its groups with `DeviceMesh.from_group()`.
*   **Keys**: The context keeps each mesh under its domain object. A domain that does not equal a built domain gets a `ValueError` from `mesh_for()`, even if its name matches. The message asks to pass the domain in `extra_domains`.
*   **Timeouts**: `set_timeout()` covers every group of the context, including those of the extra domains.

### The Contract of the Context

`DistributedContext` answers four kinds of questions. Only communication needs process groups.

| Kind | Members | In a single-process job |
|:-----|:--------|:------------------------|
| Rank | `rank`, `world_size`, `is_main_process`, `local_rank`, `is_local_main_process`, `current_device` | Rank 0 of 1 |
| Coordinates | `coordinate()`, `dim_size()` | Coordinate 0, size 1 |
| Communication | `is_distributed`, `mesh_for()`, `wait_world()`, `main_process_first()`, `local_main_process_first()`, `set_timeout()` | `is_distributed` is false, `mesh_for()` raises with a message that the job has no parallelism, `set_timeout()` does nothing, the others only wait for the GPU |
| Lifecycle | `close()`, `__enter__()`, `__exit__()` | Nothing to destroy |

`is_distributed` tells whether the job has more than one rank. This DEP introduces `coordinate()` and `dim_size()`:

```python
def coordinate(self, domain: DeviceMeshDomain[DimT], dim: DimT) -> int:
    """Returns the coordinate of this rank along a dimension of the mesh of a domain.

    Args:
        domain: A domain of this context.
        dim: The name of a dimension of the domain.

    Returns:
        The coordinate, 0 in a job without parallelism.

    Raises:
        ValueError: If the context has no such domain or the domain has no such dimension.
    """

def dim_size(self, domain: DeviceMeshDomain[DimT], dim: DimT) -> int:
    """Returns the number of ranks along a dimension of the mesh of a domain, 1 in a job without parallelism."""
```

`coordinate()` reads the coordinate from the mesh of the domain, and `dim_size()` reads the size from the dimensions of the domain. In a job without parallelism, they return 0 and 1, so code that only asks where the rank is uses them without branches. Code that communicates checks `is_distributed` before it calls `mesh_for()`.

### Closing

A context owns process groups, so it closes like a file. `DistributedContext` gets `close()` and works as a context manager.

*   **What it closes**: `close()` destroys the groups that the context created. It destroys the default process group only if the context created it.
*   **Once**: A second `close()` does nothing. After `close()`, the members that communicate raise a `RuntimeError`.
*   **Owner**: The code that calls `init_dist_context()` closes the context. Jobs take the context and never close it.
*   **On an error**: If the `with` block raises, the context marks itself closed but keeps its groups. Destroying a group waits for the other ranks, which can wait in a collective with this rank, so the job would hang until the timeout. The process exits with the error, and the launcher stops the other ranks. torchtitan also destroys its groups only after a successful run.

### Logging

`init_dist_context()` no longer touches the logging configuration. d9d modules log through `logging.getLogger(__name__)`, which are children of the `d9d` logger. `TrainingConfigurator` and `InferenceConfigurator` configure the `d9d` logger with `build_dist_logger(dist_context, level)` of `d9d.dist_context` in `configure()`. Each line shows the coordinates of the rank in the base domain.

## Usage

The configuration of a job with pipeline parallelism, HSDP inside each node and expert parallelism:

```yaml
mesh:
  pipeline_parallel: 2
  data_parallel: 16
  dense:
    data_parallel_shard: 8
  expert:
    expert_parallel: 8
```

A job with an extra domain:

```python
from enum import StrEnum

from d9d.dist_context import DeviceMeshDomain, DeviceMeshParameters, MeshDim, init_dist_context
from d9d.loop.run import TrainingConfigurator

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

with init_dist_context(config.mesh, extra_domains=(DP_CP_DOMAIN,)) as dist_context:
    trainer = TrainingConfigurator(
        dist_context=dist_context,
        # The parameters and the providers of the job.
    ).configure()
    trainer.train()
```

A provider gets the mesh with `dist_context.mesh_for(DP_CP_DOMAIN)`. Code that only needs the coordinates of the rank works without parallelism too:

```python
shard_index = dist_context.coordinate(DP_CP_DOMAIN, DataContextParallelDim.DP_CP)
num_shards = dist_context.dim_size(DP_CP_DOMAIN, DataContextParallelDim.DP_CP)
```

A job with `data_parallel: 8` on two GPUs, with the configuration unchanged:

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_MULTI_RANK_GPU_ENABLE=1 torchrun --nproc-per-node 8 train.py config.yaml
```

## Backward Compatibility

*   **Configuration**: `data_parallel_replicate` and `data_parallel_shard` become `data_parallel` and `dense.data_parallel_shard`, e.g. HSDP with `data_parallel_replicate: 2` and `data_parallel_shard: 4` becomes `data_parallel: 8` with `dense.data_parallel_shard: 4`. Context parallelism changes the same way. Pure DDP with `data_parallel_replicate: 8` becomes `data_parallel: 8`. `expert_parallel` moves to `expert.expert_parallel`, and a job with experts must set it, e.g. to 1 for experts without sharding.
*   **Building the context**: `params.build()` and `DistributedContext(params, log_level)` become `init_dist_context(params)`. The caller closes the context.
*   **Jobs**: `TrainingConfigurator(mesh=..., ...)` becomes `TrainingConfigurator(dist_context=..., ...)`, and the same for `InferenceConfigurator`. A job script creates the context in a `with` block. A script that never closes its context keeps working, with the PyTorch warning at exit as before.
*   **Domain names**: `mesh_for("dense")` becomes `mesh_for(DENSE_DOMAIN)`. Code that passes the constants needs no change. A domain typed as `str` becomes `DeviceMeshDomain`.
*   **Dimension names**: A string such as `mesh["dp_replicate"]` can stay, but `DenseDim.DP_REPLICATE` lets the type checker check it.
*   **Regular and batch domains**: `REGULAR_DOMAIN` and `BATCH_DOMAIN` become `BASE_DOMAIN`. Code that slices `dp_replicate`, `dp_shard`, `cp_shard` or `cp_replicate` from the regular mesh slices the dense mesh instead.
*   **Custom domains**: Appending to `ALL_DOMAIN_PROVIDERS` becomes `extra_domains`. A domain returns the dimensions of its mesh from `mesh_dims()` instead of building it in `build_mesh()`.
*   **Expert mesh**: It gets the dimension `tp`. A configuration where `ep` divides `dp * cp * tp` but not `dp * cp` now fails the validation.
*   **Flat domain**: `mesh_for(FLAT_DOMAIN).get_group()` becomes `mesh_for(METRIC_DOMAIN)[MetricDim.PP_DP_CP].get_group()` for reductions of the loss or the metrics, and `None` for the default group. `mesh_for(FLAT_DOMAIN).get_rank()` becomes `dist_context.rank`.
*   **Parameters**: `dist_context.mesh_params.is_distributed` becomes `dist_context.is_distributed`. `params.has_tensor_parallel` and the other `has_*` properties become `dist_context.dim_size(domain, dim) > 1` or a comparison of the degree.
*   **`current_device`**: It returns the GPU with its index. Code that compares it with `torch.device("cuda")` gets another result.
*   **Removed properties**: `master_addr` becomes `MASTER_ADDR` from the environment. `num_nodes` and `node_rank` become `WORLD_SIZE // LOCAL_WORLD_SIZE` and `GROUP_RANK` from [`torchrun`](https://docs.pytorch.org/docs/stable/elastic/run.html#environment-variables).
*   **Logging**: `dist_context.logger` and the log level of the context no longer exist. Code logs through `logging.getLogger(__name__)`. Code that uses the context without a configurator calls `build_dist_logger(dist_context, level)` itself.
*   **Profiles**: The file name of a trace holds the 4 coordinates of the base domain instead of 6.
*   **Tensor parallelism**: A job with its own tensor parallelism gets gradients and gradient norms `tp` times larger than before, i.e. correct ones.

## Alternatives Considered

### The local rank as the GPU index

torchtitan, Megatron, OLMo-core and nanotron put local rank `i` on GPU `i`. A node then cannot run more ranks than it has GPUs.

### A config field for the ranks per GPU

`DeviceMeshParameters` would get a field `ranks_per_device`, and the context would put local rank `i` on GPU `i // ranks_per_device`. But the GPU assignment depends on the machine, not on the job. A job that moves from 8 GPUs to 2 would need another configuration.

### The configurators build the context

The configurators would build the context from `mesh`, as today. Then each job owns a context, and a process with several jobs creates the communicators of the same groups again. A trainer that destroys its default group at the end also forces the next job to rendezvous again.

### Domains that build their meshes

A domain would build its mesh itself, as in the current code. A domain does not see the others, so it cannot reuse their groups. Within one mesh, it could reuse groups only through the private `DeviceMesh._flatten()` and `_unflatten()`. And `_unflatten()` creates new groups even if groups with the same ranks exist, so torchtitan, which unflattens one mesh into several views, creates duplicates.

### Meshes built by the providers

A provider would build the mesh it needs in a hook. But the model-provider hooks run once per pipeline stage, and mesh creation is collective. Several providers that need one mesh would each build it, and `set_timeout()` would miss its groups.

### Expert parallelism across tensor-parallel ranks

torchtitan splits the ranks of `dp_shard * cp * tp` into its expert dimensions, so one expert-parallel group can hold ranks of one tensor-parallel group. Without sequence parallelism, these ranks hold the same tokens. Each of them dispatches the tokens, so the experts compute every token `tp` times. d9d does not require sequence parallelism.

### A process group in a single-process job

torchtitan always creates a process group, and OLMo-core does so by default. Then `torch.distributed.is_initialized()` holds even for one process, and libraries that branch on it act as in a distributed job.

## Deferred Work

*   **Eager communicators**: With `device_id`, the default group creates its communicators at once and splits the subgroups from them with `ncclCommSplit`. So NCCL errors show up at the start, and the subgroups share resources. The memory of eager communicators on several ranks per GPU is not measured. The creation of the groups would also need its own timeout again.
*   **Arbitrary rank orders**: A domain lays out the ranks row-major, as `init_device_mesh()` does. An order such as expert parallelism along NVLink domains would need an explicit tensor of ranks in the domain. The context would find equal partitions the same way.
