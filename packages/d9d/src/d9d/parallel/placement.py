import dataclasses

from torch.distributed.tensor import DTensor, Replicate, Shard


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
