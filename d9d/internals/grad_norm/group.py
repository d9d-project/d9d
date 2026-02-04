import dataclasses
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Shard


@dataclasses.dataclass(kw_only=True, frozen=True)
class GradNormGroup:
    """
    Defines a group of parameters that share the same distributed properties.

    This grouping is used to batch gradient norm reductions efficiently. Parameters
    sharing the same device mesh shards can be reduced in a single communication collective.

    Attributes:
        shard_meshes: A tuple of device meshes where the parameters are sharded, or None if replicated/local.
        device: The device where parameters reside.
        grad_dtype: The data type of the gradients.
    """

    shard_meshes: tuple[DeviceMesh, ...] | None
    device: torch.device
    grad_dtype: torch.dtype | None


ParametersForNorm = dict[GradNormGroup, list[nn.Parameter]]


def _extract_shard_meshes(param: nn.Parameter) -> tuple[DeviceMesh, ...] | None:
    data = param.data

    if not isinstance(data, DTensor):
        return None

    mesh = data.device_mesh
    mesh_dim_names = mesh.mesh_dim_names
    if mesh_dim_names is None:
        raise ValueError("Only named meshes are supported.")

    shard_placement_dim_names: list[str] = []

    for dim_i, placement in enumerate(data.placements):
        if isinstance(placement, Shard):
            shard_placement_dim_names.append(mesh_dim_names[dim_i])

    if len(shard_placement_dim_names) == 0:
        return None

    return tuple(mesh[name] for name in shard_placement_dim_names)


def _group_sort_key(item: tuple[GradNormGroup, list[nn.Parameter]]) -> Any:
    # put items WITH shard_meshes on top so they are processed first so we benefit from comm-comp overlap
    return item[0].shard_meshes is None


def group_parameters_for_norm(parameters: Iterable[nn.Parameter]) -> ParametersForNorm:
    """
    Groups parameters based on their distributed tensor characteristics.

    Groups parameters by their sharding meshes, device, and gradient data type.

    Args:
        parameters: The iterable of parameters to group.

    Returns:
        A dictionary mapping synchronization groups to lists of parameters.
    """

    grouped_params: ParametersForNorm = defaultdict(list)
    for param in parameters:
        if not param.requires_grad:
            continue

        group = GradNormGroup(
            shard_meshes=_extract_shard_meshes(param),
            grad_dtype=param.grad_dtype,
            device=param.device
        )
        grouped_params[group].append(param)
    # we are sure dict is ordered in python 3.11 so we can sort it...
    return dict(sorted(grouped_params.items(), key=_group_sort_key))
