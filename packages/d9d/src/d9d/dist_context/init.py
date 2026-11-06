import math
import os
from collections.abc import Generator, Sequence
from contextlib import contextmanager, nullcontext
from typing import cast

import torch
import torch.distributed as dist
import torch.distributed.tensor._random
from torch.distributed import DeviceMesh

from .builtin import BASE_DOMAIN, _builtin_domains
from .configured import DistributedContext
from .domain import DeviceMeshDomain, MeshDim
from .params import DeviceMeshParameters

# The ranks of each group of one partition, sorted. Two dimensions with the same key share their groups.
_PartitionKey = tuple[tuple[int, ...], ...]


def _assign_device(local_rank: int) -> torch.device:
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise ValueError("The process sees no GPU. Check CUDA_VISIBLE_DEVICES.")
    # Without LOCAL_WORLD_SIZE, the launcher gives no count of the ranks on this node, so each rank takes its own GPU.
    if "LOCAL_WORLD_SIZE" not in os.environ or int(os.environ["LOCAL_WORLD_SIZE"]) <= num_gpus:
        if local_rank >= num_gpus:
            raise ValueError(
                f"The local rank ({local_rank}) needs its own GPU, but the process sees {num_gpus} GPUs. "
                f"Set LOCAL_WORLD_SIZE or CUDA_VISIBLE_DEVICES."
            )
        return torch.device("cuda", local_rank)
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if local_world_size % num_gpus != 0:
        raise ValueError(
            f"The ranks of this node (LOCAL_WORLD_SIZE={local_world_size}) must split evenly over "
            f"its GPUs ({num_gpus}). Change the number of ranks per node or CUDA_VISIBLE_DEVICES."
        )
    return torch.device("cuda", local_rank // (local_world_size // num_gpus))


def _check_world_size(base_dims: tuple[MeshDim, ...], world_size: int):
    # Checked before the default group exists, so a launch with the wrong number of ranks fails with this message.
    sizes = [dim.size for dim in base_dims]
    if math.prod(sizes) != world_size:
        raise ValueError(
            f"The product of the parallelism degrees ({sizes}) must equal the world size ({world_size}). "
            f"Change the degrees or the number of ranks."
        )


def _check_consistent(domain_dims: dict[DeviceMeshDomain, tuple[MeshDim, ...]], world_size: int):
    # Group creation is collective, so ranks with different domains would hang in it instead of failing.
    # Domain objects compare by identity, which does not survive pickling, so the ranks compare names and sizes.
    own = [(domain.name, [(str(dim.name), dim.size) for dim in dims]) for domain, dims in domain_dims.items()]
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, own)
    differing = [rank for rank, other in enumerate(gathered) if other != gathered[0]]
    if differing:
        raise ValueError(
            f"The ranks ({differing}) pass other domains than rank 0 ({gathered[0]}). "
            f"Pass the same params and extra_domains to init_dist_context() on every rank."
        )


def _validate(domain_dims: dict[DeviceMeshDomain, tuple[MeshDim, ...]], world_size: int):
    names = [domain.name for domain in domain_dims]
    if len(set(names)) != len(names):
        raise ValueError(f"The domain names ({names}) must be unique. Rename the extra domain.")
    for domain, dims in domain_dims.items():
        dim_names = [str(dim.name) for dim in dims]
        if len(set(dim_names)) != len(dim_names):
            raise ValueError(
                f"The dimension names ({dim_names}) of the domain ({domain.name}) must be unique. Rename a dimension."
            )
        sizes = [dim.size for dim in dims]
        if any(size < 1 for size in sizes):
            raise ValueError(
                f"The dimension sizes ({sizes}) of the domain ({domain.name}) must be positive. "
                f"Change mesh_dims() of the domain."
            )
        if math.prod(sizes) != world_size:
            raise ValueError(
                f"The dimension sizes ({sizes}) of the domain ({domain.name}) must multiply to "
                f"the world size ({world_size}). Change mesh_dims() of the domain or the parallelism degrees."
            )


def _partition(ranks: torch.Tensor, dim: int) -> _PartitionKey:
    parts = ranks.movedim(dim, -1).reshape(-1, ranks.shape[dim]).tolist()
    return tuple(sorted(tuple(part) for part in parts))


def _create_group(key: _PartitionKey, world_size: int) -> dist.ProcessGroup:
    if len(key[0]) == world_size:
        # The default group exists at this point, so WORLD is a process group.
        return cast(dist.ProcessGroup, dist.group.WORLD)
    group, _ = dist.new_subgroups_by_enumeration([list(part) for part in key])
    # GroupMember.NON_GROUP_MEMBER is -100. Every rank is in one group of a partition, so this never happens.
    if not isinstance(group, dist.ProcessGroup):
        raise RuntimeError("The current rank is in no group of its partition.")
    return group


@contextmanager
def _dtensor_rng_tracker(mesh: DeviceMesh) -> Generator[None, None, None]:
    # Without this tracker, the first DTensor random op creates one that broadcasts the generator state of rank 0
    # and overwrites the seeds of the other pipeline stages. Stopgap: PyTorch has no public API to create it without
    # the broadcast. The tracker takes only the device type and the global rank from the mesh, so any mesh with all
    # the ranks works.
    dtensor_random = torch.distributed.tensor._random  # noqa: SLF001 - no public API
    previous = dtensor_random._rng_tracker  # noqa: SLF001 - no public API
    dtensor_random._rng_tracker = dtensor_random.OffsetBasedRNGTracker(  # noqa: SLF001 - no public API
        mesh, run_state_sync=False
    )
    try:
        yield
    finally:
        dtensor_random._rng_tracker = previous  # noqa: SLF001 - no public API


def _build_meshes(
    domain_dims: dict[DeviceMeshDomain, tuple[MeshDim, ...]], world_size: int
) -> tuple[dict[DeviceMeshDomain, DeviceMesh], list[dist.ProcessGroup]]:
    groups: dict[_PartitionKey, dist.ProcessGroup] = {}
    meshes: dict[DeviceMeshDomain, DeviceMesh] = {}
    for domain, dims in domain_dims.items():
        ranks = torch.arange(world_size).reshape([dim.size for dim in dims])
        dim_groups = []
        for dim in range(ranks.ndim):
            key = _partition(ranks, dim)
            if key not in groups:
                groups[key] = _create_group(key, world_size)
            dim_groups.append(groups[key])
        meshes[domain] = DeviceMesh.from_group(
            dim_groups, "cuda", mesh=ranks, mesh_dim_names=tuple(str(dim.name) for dim in dims)
        )
    created = [group for group in groups.values() if group is not dist.group.WORLD]
    return meshes, created


@contextmanager
def init_dist_context(
    params: DeviceMeshParameters,
    extra_domains: Sequence[DeviceMeshDomain] = (),
) -> Generator[DistributedContext, None, None]:
    """Builds the distributed context of this process for the ``with`` block and sets its current GPU.

    In a distributed job, every rank must call it with the same ``params`` and the same ``extra_domains``.

    On exit, it restores the ``DTensor`` RNG tracker. If the block did not raise, it also destroys the process groups
    that it created, including the default one.

    Args:
        params: The parallelism degrees and the sections of the built-in domains.
        extra_domains: The domains to build after the built-in domains.

    Yields:
        The distributed context.

    Raises:
        ValueError: If the degrees do not match the world size or the GPUs, if a domain is invalid, or if the
            ranks pass different domains.
    """
    if dist.is_initialized():
        world_size, rank = dist.get_world_size(), dist.get_rank()
    else:
        world_size, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))

    domain_dims = {domain: domain.mesh_dims(params) for domain in (*_builtin_domains(params), *extra_domains)}
    _check_world_size(domain_dims[BASE_DOMAIN], world_size)

    local_rank = dist.get_node_local_rank(fallback_rank=0)
    device = _assign_device(local_rank)
    torch.cuda.set_device(device)

    is_distributed = world_size > 1
    owns_default_group = is_distributed and not dist.is_initialized()
    if owns_default_group:
        dist.init_process_group()

    # On an error, the groups stay: destroying them would wait for ranks that can hang in a collective.
    if is_distributed:
        _check_consistent(domain_dims, world_size)
    _validate(domain_dims, world_size)

    if is_distributed:
        meshes, groups = _build_meshes(domain_dims, world_size)
    else:
        meshes, groups = {}, []

    with _dtensor_rng_tracker(meshes[BASE_DOMAIN]) if is_distributed else nullcontext():
        yield DistributedContext(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            domain_dims=domain_dims,
            meshes=meshes,
            groups=groups,
        )

    # Reached only if the block did not raise: destroying a group waits for ranks that can hang in a collective.
    # The script can call destroy_process_group(), which destroys all groups.
    if not dist.is_initialized():
        return
    # Reverse order, as destroy_process_group() does: ncclCommAbort() is collective in some NCCL versions.
    for group in reversed(groups):
        dist.destroy_process_group(group)
    if owns_default_group:
        dist.destroy_process_group()
