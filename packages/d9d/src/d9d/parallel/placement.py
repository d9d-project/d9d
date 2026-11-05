import dataclasses

from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, Replicate, Shard, distribute_tensor


@dataclasses.dataclass(kw_only=True, frozen=True)
class PlacementDims:
    """Mesh dimensions of a ``DTensor`` parameter, split by placement.

    Attributes:
        replicate_dim_names: The names of the mesh dimensions with a ``Replicate`` placement.
        shard_dim_names: The names of the mesh dimensions with a ``Shard`` placement.
    """

    replicate_dim_names: tuple[str, ...]
    shard_dim_names: tuple[str, ...]


def split_placement_dims(data: DTensor) -> PlacementDims:
    """Splits the mesh dimensions of a ``DTensor`` parameter into replicated and sharded ones.

    Args:
        data: The parameter tensor.

    Returns:
        The mesh dimensions by placement.

    Raises:
        ValueError: If the device mesh has no dimension names, or if a placement is neither ``Replicate`` nor
            ``Shard``.
    """
    mesh_dim_names = data.device_mesh.mesh_dim_names
    if mesh_dim_names is None:
        raise ValueError("The device mesh of a parameter has no dimension names. Build the mesh with mesh_dim_names.")

    replicate_dim_names: list[str] = []
    shard_dim_names: list[str] = []
    for dim_name, placement in zip(mesh_dim_names, data.placements, strict=True):
        match placement:
            case Replicate():
                replicate_dim_names.append(dim_name)
            case Shard():
                shard_dim_names.append(dim_name)
            case _:
                raise ValueError(f"Parameter placement ({placement}) is not supported. Use Replicate or Shard.")

    return PlacementDims(replicate_dim_names=tuple(replicate_dim_names), shard_dim_names=tuple(shard_dim_names))


def distribute_parameters(module: nn.Module, mesh: DeviceMesh, placements: tuple[Placement, ...]):
    """Turns the parameters of a module into ``DTensor`` objects without changing how the module runs.

    A parameter that several submodules hold stays one parameter.

    Args:
        module: The module.
        mesh: The device mesh.
        placements: The placements of the parameters, one per mesh dimension.
    """
    distributed: dict[int, nn.Parameter] = {}
    for submodule in module.modules():
        for name, param in list(submodule.named_parameters(recurse=False, remove_duplicate=False)):
            if id(param) not in distributed:
                data = distribute_tensor(param.data, mesh, placements)
                distributed[id(param)] = nn.Parameter(data, requires_grad=param.requires_grad)
            submodule.register_parameter(name, distributed[id(param)])
