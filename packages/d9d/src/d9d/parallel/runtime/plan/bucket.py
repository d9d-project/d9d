"""The buckets of a stage: the roles of the mesh dimensions and the bucket of each parameter."""

import dataclasses
import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.distributed import DeviceMesh, ProcessGroup
from torch.distributed.tensor import DTensor, Shard

from d9d.parallel.placement import split_placement_dims

from .parameter import ParameterPlan, local_tensor


@dataclasses.dataclass(frozen=True, slots=True)
class DimRoles:
    """The roles of the mesh dimensions of a parameter.

    A mesh dimension that neither gathers nor replicates the parameter shards it for its layer, as expert parallelism
    does. The runtime leaves it alone.

    Attributes:
        gather: The mesh dimension that the runtime gathers before compute, or ``None``.
        replicas: The mesh dimensions over which the runtime sums the gradients.
    """

    gather: str | None
    replicas: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class BucketKey:
    """The properties that the parameters of one bucket share.

    Attributes:
        mesh: The device mesh of the parameters, or ``None`` for parameters outside a device mesh.
        roles: The roles of the mesh dimensions.
        dtype: The dtype of the weights.
        grad_dtype: The dtype of the gradients, or ``None`` in a frozen bucket.
        groups: The groups that hold the parameters, or empty without a gather.
        trainable: Whether the bucket computes gradients.
    """

    mesh: DeviceMesh | None
    roles: DimRoles
    dtype: torch.dtype
    grad_dtype: torch.dtype | None
    groups: tuple[int, ...]
    trainable: bool


@dataclasses.dataclass(slots=True)
class GatherPlan:
    """The plan of the all-gather of a bucket.

    Attributes:
        process_group: The process group of the gather dimension.
        shard_sizes: The number of elements of the shard of each rank without padding, one list per parameter.
    """

    process_group: ProcessGroup
    shard_sizes: list[list[int]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class BucketPlan:
    """The plan of one bucket: parameters that share their collectives and buffers.

    Attributes:
        parameters: The parameters, in buffer order.
        groups: The groups that hold the parameters, in ascending order.
        gather: The gather of the bucket, or ``None`` for a bucket without a gather.
        replica_process_groups: The process groups of the replica dimensions, in reduction order.
        dtype: The dtype of the weights.
        grad_dtype: The dtype of the gradients, or ``None`` in a frozen bucket.
        trainable: Whether the bucket computes gradients.
        numel: The number of elements of the buffers, including the padding of uneven chunks.
    """

    parameters: list[ParameterPlan]
    groups: tuple[int, ...]
    gather: GatherPlan | None
    replica_process_groups: list[ProcessGroup]
    dtype: torch.dtype
    grad_dtype: torch.dtype | None
    trainable: bool
    numel: int = 0


def chunk_places(gather: GatherPlan, flat: Tensor) -> list[Tensor]:
    """Returns the place of each parameter in a flat buffer of a gathered bucket.

    Args:
        gather: The gather of the bucket.
        flat: A buffer with the padded chunks of all parameters.

    Returns:
        The place of each parameter: its padded chunks of all ranks, next to each other.
    """
    parts = gather.process_group.size()
    places, offset = [], 0
    for sizes in gather.shard_sizes:
        width = sizes[0]
        places.append(flat[parts * offset : parts * (offset + width)])
        offset += width
    return places


def classify_dims(parameter: ParameterPlan, data: DTensor, gather_dims: frozenset[str]) -> DimRoles:
    """Assigns a role to each mesh dimension of a parameter.

    Args:
        parameter: The parameter, whose name the errors report.
        data: The data of the sharded parameter.
        gather_dims: The mesh dimensions along which data parallelism shards parameters.

    Returns:
        The roles of the mesh dimensions.

    Raises:
        ValueError: If the placements shard the parameter along several gather dimensions, not with ``Shard(0)``, or
            along a gather dimension and another one.
    """
    dims = split_placement_dims(data)
    names = cast(tuple[str, ...], data.device_mesh.mesh_dim_names)
    gathered = [name for name in dims.shard_dim_names if name in gather_dims]

    if len(gathered) > 1 or any(data.placements[names.index(name)] != Shard(0) for name in gathered):
        raise ValueError(
            f"The placements ({data.placements}) of the parameter ({parameter.name}) shard it along the gather "
            f"dimensions ({gathered}) in a way the runtime cannot gather. Use Shard(0) along one gather dimension."
        )
    if gathered and len(dims.shard_dim_names) > 1:
        # The gather would rebuild the global row count, but the layer needs only its own part.
        raise ValueError(
            f"The placements ({data.placements}) of the parameter ({parameter.name}) shard it along a gather "
            "dimension and along another mesh dimension. Shard it along gather dimensions or other ones, not both."
        )

    return DimRoles(gather=gathered[0] if gathered else None, replicas=dims.replicate_dim_names)


def _bucket_key(parameter: ParameterPlan, groups: tuple[int, ...], gather_dims: frozenset[str]) -> BucketKey:
    data = parameter.sharded.data
    mesh, roles = None, DimRoles(gather=None, replicas=())
    if isinstance(data, DTensor):
        mesh, roles = data.device_mesh, classify_dims(parameter, data, gather_dims)
    return BucketKey(
        mesh=mesh,
        roles=roles,
        dtype=parameter.sharded.dtype,
        # The gradient dtype of a frozen parameter does not matter, so frozen parameters share buckets regardless.
        grad_dtype=parameter.sharded.grad_dtype if parameter.trainable else None,
        # Gathered weights live only in their groups. Weights without a gather live all the time.
        groups=() if roles.gather is None else groups,
        trainable=parameter.trainable,
    )


def _new_bucket(key: BucketKey, groups: tuple[int, ...]) -> BucketPlan:
    gather, replica_process_groups = None, []
    if key.mesh is not None:
        if key.roles.gather is not None:
            gather = GatherPlan(process_group=key.mesh[key.roles.gather].get_group())
        if key.roles.replicas:
            # From the innermost mesh dimension to the outermost, the order of the gradient synchronizer, so the sums
            # round the same way.
            replica_process_groups = key.mesh[key.roles.replicas].get_all_groups()[::-1]

    return BucketPlan(
        parameters=[],
        groups=groups,
        gather=gather,
        replica_process_groups=replica_process_groups,
        dtype=key.dtype,
        grad_dtype=key.grad_dtype,
        trainable=key.trainable,
    )


def plan_buckets(
    parameters_by_group: list[list[ParameterPlan]], bucket_size_mib: int, gather_dims: frozenset[str]
) -> list[BucketPlan]:
    """Groups the parameters of a stage into buckets.

    Args:
        parameters_by_group: The parameters that each group holds, by group index.
        bucket_size_mib: The size limit of buckets without a gather, in MiB.
        gather_dims: The mesh dimensions along which data parallelism shards parameters.

    Returns:
        The buckets in reduction order: the buckets of the last group first.

    Raises:
        ValueError: If the placements of a parameter are not supported, or if the shard of a gathered parameter does
            not match ``torch.chunk``.
    """
    uses: dict[ParameterPlan, list[int]] = {}
    for group_index, group_parameters in enumerate(parameters_by_group):
        for parameter in group_parameters:
            uses.setdefault(parameter, []).append(group_index)

    buckets: list[BucketPlan] = []
    open_buckets: dict[BucketKey, BucketPlan] = {}
    max_bytes = bucket_size_mib * 1024 * 1024

    for group_index in reversed(range(len(parameters_by_group))):
        for parameter in reversed(parameters_by_group[group_index]):
            groups = tuple(uses[parameter])
            # A tied parameter joins a bucket once, in the last group that uses it.
            if groups[-1] != group_index:
                continue

            key = _bucket_key(parameter, groups, gather_dims)
            gathered = key.roles.gather is not None
            # Without a gather or gradients, the runtime has nothing to do for a parameter.
            if not gathered and not parameter.trainable:
                continue

            size = local_tensor(parameter.sharded.data).numel()
            bucket = open_buckets.get(key)
            # A bucket without a gather is trainable, and plan_parameters() rejects a trainable parameter without
            # grad_dtype.
            if bucket is None or (
                not gathered and (bucket.numel + size) * cast(torch.dtype, key.grad_dtype).itemsize > max_bytes
            ):
                bucket = _new_bucket(key, groups)
                open_buckets[key] = bucket
                buckets.append(bucket)

            if bucket.gather is not None:
                size = _add_shard_sizes(bucket.gather, parameter)

            bucket.parameters.append(parameter)
            bucket.numel += size
            bucket.groups = tuple(sorted({*bucket.groups, *groups}))

    return buckets


def _add_shard_sizes(gather: GatherPlan, parameter: ParameterPlan) -> int:
    process_group = gather.process_group
    parts = process_group.size()
    sizes = shard_sizes(parameter.sharded, parts)
    local = local_tensor(parameter.sharded.data).numel()
    if local != sizes[process_group.rank()]:
        raise ValueError(
            f"The shard of the parameter ({parameter.name}) has ({local}) elements, but torch.chunk gives "
            f"({sizes[process_group.rank()]}). Distribute the parameter with distribute_tensor(), whose Shard(0) "
            "chunks follow torch.chunk."
        )
    gather.shard_sizes.append(sizes)

    # Every rank holds a padded chunk of equal size, so the collectives need no uneven variants.
    return parts * sizes[0]


def shard_sizes(sharded: nn.Parameter, parts: int) -> list[int]:
    """Returns the number of elements of the ``Shard(0)`` shard of each rank, without padding.

    Args:
        sharded: The sharded parameter.
        parts: The number of ranks of the gather dimension.

    Returns:
        The number of elements of each shard, by rank.
    """
    # Matches a DTensor with Shard(0), which follows torch.chunk: full chunks first, so the last ones are smaller.
    rows, row_size = sharded.shape[0], math.prod(sharded.shape[1:])
    full = math.ceil(rows / parts)
    return [max(0, min(full, rows - full * rank)) * row_size for rank in range(parts)]
