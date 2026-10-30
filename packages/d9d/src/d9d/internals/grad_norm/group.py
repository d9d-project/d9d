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
    """Key of a group of parameters with the same distributed properties.

    Parameters sharded over the same device meshes share one collective for their gradient norm.

    Attributes:
        shard_meshes: The device meshes the parameters are sharded over, or ``None`` if they are replicated or
            local.
        device: The device of the parameters.
        grad_dtype: The dtype of the gradients.
    """

    shard_meshes: tuple[DeviceMesh, ...] | None
    device: torch.device
    grad_dtype: torch.dtype | None


ParametersForNorm = dict[GradNormGroup, list[nn.Parameter]]
"""Parameters grouped by ``GradNormGroup``, as built by ``group_parameters_for_norm``."""


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
    # Sharded groups come first, so their norm all-reduce overlaps with the norm computation of the other groups.
    return item[0].shard_meshes is None


def group_parameters_for_norm(parameters: Iterable[nn.Parameter]) -> ParametersForNorm:
    """Groups parameters by their shard meshes, device and gradient dtype.

    Parameters that do not require gradients are skipped. Groups of sharded parameters come first.

    Args:
        parameters: The parameters to group.

    Returns:
        A dict that maps groups to their parameters.
    """
    grouped_params: ParametersForNorm = defaultdict(list)
    for param in parameters:
        if not param.requires_grad:
            continue

        group = GradNormGroup(
            shard_meshes=_extract_shard_meshes(param), grad_dtype=param.grad_dtype, device=param.device
        )
        grouped_params[group].append(param)
    return dict(sorted(grouped_params.items(), key=_group_sort_key))
