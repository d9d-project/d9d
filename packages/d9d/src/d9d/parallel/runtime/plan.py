"""Planning of buckets: axis roles, bucket membership and memory requirements."""

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import cast

import torch
from torch import nn
from torch.distributed import DeviceMesh, ProcessGroup
from torch.distributed.tensor import DTensor, Shard

from d9d.parallel.placement import split_placement_dims
from d9d.pipelining.api import StageSegment, as_segments

from .parameter import ParameterPlan, local_tensor, plan_parameters


@dataclasses.dataclass(frozen=True, slots=True)
class AxisRoles:
    """Communication roles of the mesh dimensions of a parameter.

    A mesh dimension that is neither the gather axis nor a replica axis shards the parameter for its layer, as
    expert parallelism does. The runtime leaves it alone.

    Attributes:
        gather: The axis gathered before compute, or ``None``.
        replicas: The axes over which gradients are summed.
    """

    gather: str | None
    replicas: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class BucketKey:
    """Properties that parameters must share to use one bucket.

    Attributes:
        mesh: The parameter mesh, or ``None`` for a parameter outside a device mesh.
        axes: The gather and replica axes.
        grad_dtype: The gradient buffer dtype.
        segments: The segments that use the gathered weights, or empty without a gather.
        trainable: Whether the bucket computes gradients.
        compute_dtype: The dtype of the unsharded weights.
    """

    mesh: DeviceMesh | None
    axes: AxisRoles
    grad_dtype: torch.dtype
    segments: tuple[int, ...]
    trainable: bool
    compute_dtype: torch.dtype


@dataclasses.dataclass(slots=True)
class BucketPlan:
    """Parameters that share communication and gradient buffers, before the buffers exist.

    Attributes:
        parameters: The parameters, in buffer order.
        segments: The segments that use these parameters, in ascending order.
        gather_group: The process group of the gather axis, or ``None`` without a gather.
        replica_groups: The process groups of the replica axes, in reduction order.
        usage_groups: The process groups of every mesh dimension. A parameter has a gradient if any rank used any
            part of it, also along the dimensions that its layer shards.
        compute_dtype: The dtype of the unsharded weights.
        grad_dtype: The gradient buffer dtype.
        trainable: Whether the bucket computes gradients.
        shard_sizes: The unpadded element count of the shard of each rank of the gather axis, one list per
            parameter. Empty without a gather.
        numel: The buffer element count, including the padding of uneven shards.
        gradient_bytes: The gradient buffer size, or 0 without gradients.
    """

    parameters: list[ParameterPlan]
    segments: tuple[int, ...]
    gather_group: ProcessGroup | None
    replica_groups: list[ProcessGroup]
    usage_groups: list[ProcessGroup]
    compute_dtype: torch.dtype
    grad_dtype: torch.dtype
    trainable: bool
    shard_sizes: list[list[int]] = dataclasses.field(default_factory=list)
    numel: int = 0
    gradient_bytes: int = 0

    @property
    def gathers(self) -> bool:
        """Whether the bucket gathers its weights before use."""
        return self.gather_group is not None


@dataclasses.dataclass(slots=True)
class StagePlan:
    """Parameters and buckets of one stage, before the buffers exist.

    Attributes:
        segments: The segments of the stage, in forward order.
        parameters: The parameters of the stage, each once.
        parameters_by_segment: The parameters that each segment uses, by segment index.
        buckets: The buckets in reduction order.
    """

    segments: Sequence[StageSegment]
    parameters: list[ParameterPlan]
    parameters_by_segment: list[list[ParameterPlan]]
    buckets: list[BucketPlan]


def classify_axes(data: DTensor, gather_dims: frozenset[str]) -> AxisRoles:
    """Assigns a communication role to each mesh dimension of a parameter.

    Args:
        data: The data of the sharded parameter.
        gather_dims: The mesh dimensions along which data parallelism shards parameters.

    Returns:
        The gather axis and the replica axes.

    Raises:
        ValueError: If the placements are not supported.
    """
    dims = split_placement_dims(data)
    names = cast(tuple[str, ...], data.device_mesh.mesh_dim_names)
    gathered = [name for name in dims.shard_dim_names if name in gather_dims]

    if len(gathered) > 1 or any(data.placements[names.index(name)] != Shard(0) for name in gathered):
        raise ValueError(
            f"The placements ({data.placements}) shard along the data shard dimensions ({gathered}) in a way the "
            "parameter runtime cannot gather. Use Shard(0) along one data shard dimension."
        )
    if gathered and len(dims.shard_dim_names) > 1:
        # The gather would rebuild the global row count, but the layer needs only its own part.
        raise ValueError(
            f"The placements ({data.placements}) shard a gathered parameter along another dimension too. "
            "Gather such parameters in separate modules."
        )

    return AxisRoles(gather=gathered[0] if gathered else None, replicas=dims.replicate_dim_names)


def _bucket_key(parameter: ParameterPlan, segments: tuple[int, ...], gather_dims: frozenset[str]) -> BucketKey:
    data = parameter.sharded.data
    if not isinstance(data, DTensor):
        return BucketKey(
            mesh=None,
            axes=AxisRoles(gather=None, replicas=()),
            grad_dtype=parameter.grad_dtype,
            segments=(),
            trainable=parameter.trainable,
            compute_dtype=parameter.compute_dtype,
        )

    axes = classify_axes(data, gather_dims)
    # Gathered weights live only in their segments. Weights without a gather live all the time.
    return BucketKey(
        mesh=data.device_mesh,
        axes=axes,
        grad_dtype=parameter.grad_dtype,
        segments=() if axes.gather is None else segments,
        trainable=parameter.trainable,
        compute_dtype=parameter.compute_dtype,
    )


def _new_bucket(key: BucketKey, segments: tuple[int, ...]) -> BucketPlan:
    gather_group, replica_groups, usage_groups = None, [], []
    if key.mesh is not None:
        usage_groups = key.mesh.get_all_groups()
        if key.axes.gather is not None:
            gather_group = key.mesh[key.axes.gather].get_group()
        if key.axes.replicas:
            # From the innermost mesh dimension to the outermost, the order of the earlier gradient synchronizer,
            # so the sums round the same way.
            replica_groups = key.mesh[key.axes.replicas].get_all_groups()[::-1]

    return BucketPlan(
        parameters=[],
        segments=segments,
        gather_group=gather_group,
        replica_groups=replica_groups,
        usage_groups=usage_groups,
        compute_dtype=key.compute_dtype,
        grad_dtype=key.grad_dtype,
        trainable=key.trainable,
    )


def plan_buckets(
    parameters_by_segment: list[list[ParameterPlan]], bucket_size_mib: int, gather_dims: frozenset[str]
) -> list[BucketPlan]:
    """Groups the parameters of a stage into buckets.

    Args:
        parameters_by_segment: The parameters that each segment uses.
        bucket_size_mib: The size limit of buckets without a gather, in MiB.
        gather_dims: The mesh dimensions along which data parallelism shards parameters.

    Returns:
        The buckets in backward order. The order depends only on the model, so collectives match across ranks.

    Raises:
        ValueError: If the shard of a gathered parameter does not match ``torch.chunk``.
    """
    uses: dict[int, list[int]] = {}
    for segment_index, segment_parameters in enumerate(parameters_by_segment):
        for parameter in segment_parameters:
            uses.setdefault(id(parameter), []).append(segment_index)

    buckets: list[BucketPlan] = []
    open_buckets: dict[BucketKey, BucketPlan] = {}
    max_bytes = bucket_size_mib * 1024 * 1024

    # The backward pass runs the segments in reverse order, so a bucket fills up in that order.
    for segment_index in reversed(range(len(parameters_by_segment))):
        for parameter in reversed(parameters_by_segment[segment_index]):
            segments = tuple(uses[id(parameter)])
            # A tied parameter joins a bucket once, in the last segment that uses it.
            if segments[-1] != segment_index:
                continue

            key = _bucket_key(parameter, segments, gather_dims)
            gathered = key.axes.gather is not None
            # Without a gather or gradients, the runtime has nothing to do for a parameter.
            if not gathered and not parameter.trainable:
                continue

            size = local_tensor(parameter.sharded.data).numel()
            bucket = open_buckets.get(key)
            if bucket is None or (not gathered and bucket.gradient_bytes + size * key.grad_dtype.itemsize > max_bytes):
                bucket = _new_bucket(key, segments)
                open_buckets[key] = bucket
                buckets.append(bucket)

            if bucket.gather_group is not None:
                size = _add_shard_sizes(bucket, bucket.gather_group, parameter)

            bucket.parameters.append(parameter)
            bucket.numel += size
            if parameter.trainable:
                bucket.gradient_bytes += size * key.grad_dtype.itemsize
            bucket.segments = tuple(sorted({*bucket.segments, *segments}))

    return buckets


def _add_shard_sizes(bucket: BucketPlan, gather_group: ProcessGroup, parameter: ParameterPlan) -> int:
    parts = gather_group.size()
    sizes = shard_sizes(parameter.sharded, parts)
    if local_tensor(parameter.sharded.data).numel() != sizes[gather_group.rank()]:
        raise ValueError(
            f"The shard of ({parameter.owners[0][1]}) does not match torch.chunk. "
            "Distribute the parameter with parallelize_fsdp() or parallelize_hsdp()."
        )
    bucket.shard_sizes.append(sizes)

    # Every rank holds a padded row of equal size, so the collectives need no uneven variants.
    return parts * sizes[0]


def shard_sizes(sharded: nn.Parameter, parts: int) -> list[int]:
    """Returns the unpadded element count of each ``Shard(0)`` chunk.

    Args:
        sharded: The sharded parameter.
        parts: The number of ranks of the gather axis.

    Returns:
        The element count of the chunk of each rank.
    """
    # Matches a DTensor with Shard(0), which follows torch.chunk: full chunks first, so the last ones are smaller.
    rows, row_size = sharded.shape[0], math.prod(sharded.shape[1:])
    full = math.ceil(rows / parts)
    return [max(0, min(full, rows - full * rank)) * row_size for rank in range(parts)]


def minimum_stage_budget(buckets: Sequence[BucketPlan]) -> int:
    """Returns the floor of the memory budget of a stage, a conservative estimate for its gathered buckets.

    Args:
        buckets: The buckets of the stage.

    Returns:
        The budget in bytes.
    """
    weights: dict[int, int] = {}
    gradients: dict[int, int] = {}
    for bucket in buckets:
        if not bucket.gathers:
            continue

        size = bucket.numel * bucket.compute_dtype.itemsize
        for segment in bucket.segments:
            weights[segment] = weights.get(segment, 0) + size
        # Tied gradients stay live between their last and first uses, including segments that do not use them.
        for segment in range(bucket.segments[0], bucket.segments[-1] + 1):
            gradients[segment] = gradients.get(segment, 0) + bucket.gradient_bytes

    # The current segment, the prefetch of the next one, and the inputs and outputs of collectives. The factor is a
    # stopgap: the runtime does not yet measure the peak of a segment.
    return 4 * (max(weights.values(), default=0) + max(gradients.values(), default=0))


def plan_stages(
    modules: Mapping[int, nn.Module],
    compute_dtype: torch.dtype | None,
    training: bool,
    bucket_size_mib: int,
    gather_dims: frozenset[str],
) -> dict[int, StagePlan]:
    """Plans the buckets of every stage before any buffer exists.

    Args:
        modules: The stage modules, by stage index.
        compute_dtype: The dtype of unsharded parameters, or ``None`` to keep the dtype of the storage.
        training: Whether the job computes gradients.
        bucket_size_mib: The size limit of buckets without a gather, in MiB.
        gather_dims: The mesh dimensions along which data parallelism shards parameters.

    Returns:
        The plan of each stage.

    Raises:
        ValueError: If a stage has no segments, if a parameter belongs to no segment, if a parameter is shared by
            several stages, or if a shard does not match ``torch.chunk``.
    """
    stages: dict[int, StagePlan] = {}
    parameter_stages: dict[int, int] = {}

    for stage_index, module in modules.items():
        segments = as_segments(module).stage_segments()
        check_segments(module, segments)

        parameters_by_segment = plan_parameters(module, segments, compute_dtype, training)
        parameters = list({id(x): x for segment in parameters_by_segment for x in segment}.values())
        for parameter in parameters:
            owner = parameter_stages.setdefault(id(parameter.sharded), stage_index)
            if owner != stage_index:
                raise ValueError(
                    f"A parameter is shared by stages ({owner}) and ({stage_index}). "
                    "Place tied parameters on one stage."
                )

        stages[stage_index] = StagePlan(
            segments=segments,
            parameters=parameters,
            parameters_by_segment=parameters_by_segment,
            buckets=plan_buckets(parameters_by_segment, bucket_size_mib, gather_dims),
        )

    return stages


def check_segments(module: nn.Module, segments: Sequence[StageSegment]):
    """Checks that a stage has segments and that they hold every parameter of the module.

    Args:
        module: The stage module.
        segments: The segments of the stage.

    Raises:
        ValueError: If the stage has no segments, or if a parameter belongs to no segment.
    """
    if not segments:
        raise ValueError(f"The stage module ({type(module).__name__}) has no segments. Return at least one.")

    # Every parameter needs its unsharded value during the forward pass, including frozen weights.
    owned = {id(p) for segment in segments for p in segment.module.parameters()}
    missing = [name for name, p in module.named_parameters() if id(p) not in owned]
    if missing:
        raise ValueError(f"The parameters ({missing}) belong to no segment. Add their modules to a segment.")
