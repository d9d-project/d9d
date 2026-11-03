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
from d9d.pipelining.api import as_segments

from .parameter import ParameterHandle, build_handles


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
        handles: The parameters, in buffer order.
        segments: The segments that use these parameters, in ascending order.
        gather_group: The process group of the gather axis, or ``None`` without a gather.
        replica_groups: The process groups of the replica axes, in reduction order.
        grad_dtype: The gradient buffer dtype.
        trainable: Whether the bucket computes gradients.
        numel: The buffer element count, including the padding of uneven shards.
        gradient_bytes: The gradient buffer size, or 0 without gradients.
    """

    handles: list[ParameterHandle]
    segments: tuple[int, ...]
    gather_group: ProcessGroup | None
    replica_groups: list[ProcessGroup]
    grad_dtype: torch.dtype
    trainable: bool
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
        handles: The parameters that each segment uses, by segment index.
        buckets: The buckets in reduction order.
    """

    handles: list[list[ParameterHandle]]
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


def _bucket_key(handle: ParameterHandle, segments: tuple[int, ...], gather_dims: frozenset[str]) -> BucketKey:
    data = handle.sharded.data
    grad_dtype = handle.unsharded.grad_dtype or handle.unsharded.dtype
    if not isinstance(data, DTensor):
        return BucketKey(None, AxisRoles(None, ()), grad_dtype, (), handle.trainable, handle.unsharded.dtype)
    axes = classify_axes(data, gather_dims)
    # Gathered weights live only in their segments. Weights without a gather live all the time.
    return BucketKey(
        mesh=data.device_mesh,
        axes=axes,
        grad_dtype=grad_dtype,
        segments=() if axes.gather is None else segments,
        trainable=handle.trainable,
        compute_dtype=handle.unsharded.dtype,
    )


def _new_bucket(key: BucketKey, segments: tuple[int, ...]) -> BucketPlan:
    gather_group, replica_groups = None, []
    if key.mesh is not None:
        if key.axes.gather is not None:
            gather_group = key.mesh[key.axes.gather].get_group()
        if key.axes.replicas:
            replica_groups = key.mesh[key.axes.replicas].get_all_groups()[::-1]
    return BucketPlan([], segments, gather_group, replica_groups, key.grad_dtype, key.trainable)


def plan_buckets(
    handles: list[list[ParameterHandle]], bucket_size_mb: int, gather_dims: frozenset[str]
) -> list[BucketPlan]:
    """Groups the parameters of a stage into buckets.

    Args:
        handles: The parameters that each segment uses.
        bucket_size_mb: The size limit of buckets without a gather, in MiB.
        gather_dims: The mesh dimensions along which data parallelism shards parameters.

    Returns:
        The buckets in backward order. The order depends only on the model, so collectives match across ranks.
    """
    uses: dict[int, list[int]] = {}
    for segment_index, segment_handles in enumerate(handles):
        for handle in segment_handles:
            uses.setdefault(id(handle), []).append(segment_index)
    buckets: list[BucketPlan] = []
    open_buckets: dict[BucketKey, BucketPlan] = {}
    max_bytes = bucket_size_mb * 1024 * 1024
    # The backward pass runs the segments in reverse order, so a bucket fills up in that order.
    for segment_index in reversed(range(len(handles))):
        for handle in reversed(handles[segment_index]):
            segments = tuple(uses[id(handle)])
            # A tied parameter joins a bucket once, in the last segment that uses it.
            if segments[-1] != segment_index:
                continue
            key = _bucket_key(handle, segments, gather_dims)
            gathered = key.axes.gather is not None
            # Without a gather or gradients, the runtime has nothing to do for a parameter.
            if not gathered and not handle.trainable:
                continue
            size = handle.unsharded.numel()
            bucket = open_buckets.get(key)
            if bucket is None or (not gathered and bucket.gradient_bytes + size * key.grad_dtype.itemsize > max_bytes):
                bucket = _new_bucket(key, segments)
                open_buckets[key] = bucket
                buckets.append(bucket)
            if bucket.gather_group is not None:
                # Every rank holds a padded row of equal size, so the collectives need no uneven variants.
                parts = bucket.gather_group.size()
                size = parts * shard_sizes(handle.sharded, parts)[0]
            bucket.handles.append(handle)
            bucket.numel += size
            bucket.gradient_bytes += size * key.grad_dtype.itemsize if handle.trainable else 0
            bucket.segments = tuple(sorted({*bucket.segments, *segments}))
    return buckets


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
    full = -(-rows // parts)
    return [max(0, min(full, rows - full * rank)) * row_size for rank in range(parts)]


def minimum_stage_budget(buckets: Sequence[BucketPlan]) -> int:
    """Returns the smallest memory budget that runs the gathered buckets of a stage.

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
        size = bucket.numel * bucket.handles[0].unsharded.dtype.itemsize
        for segment in bucket.segments:
            weights[segment] = weights.get(segment, 0) + size
        # Tied gradients stay live between their last and first uses, including segments that do not use them.
        for segment in range(bucket.segments[0], bucket.segments[-1] + 1):
            gradients[segment] = gradients.get(segment, 0) + bucket.gradient_bytes
    # A rough bound: the current segment, the prefetch of the next one, and the inputs and outputs of collectives.
    return 4 * (max(weights.values(), default=0) + max(gradients.values(), default=0))


def plan_stages(
    modules: Mapping[int, nn.Module],
    compute_dtype: torch.dtype | None,
    training: bool,
    bucket_size_mb: int,
    gather_dims: frozenset[str],
) -> dict[int, StagePlan]:
    """Plans the buckets of every stage before any buffer exists.

    Args:
        modules: The stage modules, by stage index.
        compute_dtype: The dtype of unsharded parameters, or ``None`` to keep the dtype of the storage.
        training: Whether the job computes gradients.
        bucket_size_mb: The size limit of buckets without a gather, in MiB.
        gather_dims: The mesh dimensions along which data parallelism shards parameters.

    Returns:
        The plan of each stage.

    Raises:
        ValueError: If a parameter is shared by several stages.
    """
    stages: dict[int, StagePlan] = {}
    parameter_stages: dict[int, int] = {}
    for stage_index, module in modules.items():
        handles = build_handles(as_segments(module).stage_segments(), compute_dtype, training)
        for segment_handles in handles:
            for handle in segment_handles:
                owner = parameter_stages.setdefault(id(handle.sharded), stage_index)
                if owner != stage_index:
                    raise ValueError(
                        f"A parameter is shared by stages ({owner}) and ({stage_index}). "
                        "Place tied parameters on one stage."
                    )
        stages[stage_index] = StagePlan(handles, plan_buckets(handles, bucket_size_mb, gather_dims))
    return stages
