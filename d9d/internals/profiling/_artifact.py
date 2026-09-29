import tarfile
from pathlib import Path

from d9d.core.dist_context import REGULAR_DOMAIN, DistributedContext


def rank_artifact_path(save_dir: Path, dist_context: DistributedContext, kind: str) -> Path:
    """Builds a per-rank JSON artifact path that is unique across the device mesh.

    Args:
        save_dir: Directory the artifact is placed into.
        dist_context: The distributed context.
        kind: Artifact kind appended to the file name (e.g. ``trace``).

    Returns:
        Path of the JSON artifact file.

    Raises:
        RuntimeError: If the current rank has no coordinate in the regular mesh.
    """
    if not dist_context.mesh_params.is_distributed:
        return save_dir / f"{kind}.json"

    mesh_regular = dist_context.mesh_for(REGULAR_DOMAIN)
    coord = mesh_regular.get_coordinate()
    if coord is None:
        raise RuntimeError("Invalid mesh")
    coord_str = "-".join(str(x) for x in coord)
    rank = mesh_regular.get_rank()
    return save_dir / f"rank-{rank}-coord-{coord_str}-{kind}.json"


def archive_artifact(path: Path) -> None:
    """Compresses an artifact into a ``.tar.gz`` archive next to it and removes the original file.

    Args:
        path: The artifact file to compress.
    """
    with tarfile.open(path.with_suffix(".tar.gz"), "w:gz") as tar:
        tar.add(path, arcname=path.name)
    path.unlink()
