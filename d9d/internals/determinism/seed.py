import os
import random
from typing import cast

import numpy as np
import torch
import torch.distributed.tensor

from d9d.core.dist_context import REGULAR_DOMAIN, DistributedContext


def set_seeds(
    dist_context: DistributedContext,
    seed: int,
    distinct_seed_mesh_dim: str = "pp",
) -> None:
    """
    Sets random seeds for Python, NumPy, and PyTorch.

    This function sets seeds deterministically based on the provided base seed and the
    process's rank within a specific mesh dimension.

    The seed is shifted by the rank in the `distinct_seed_mesh_dim` (e.g., Pipeline Parallel rank).
    This ensures that processes in different pipeline stages operate with different random states,
    while processes that should share randomness (like Expert Parallel peers) can be synchronized.

    Args:
        dist_context: The distributed context.
        seed: The base random seed.
        distinct_seed_mesh_dim: The name of the mesh dimension along which seeds should
            be distinct (e.g., 'pp' for pipeline parallelism). Ranks along other dimensions
            will share the seed.
    """

    # Mutate seed based on PP rank if distributed
    if dist_context.mesh_params.is_distributed:
        distinct_mesh = dist_context.mesh_for(REGULAR_DOMAIN)[distinct_seed_mesh_dim]
        seed = (seed + distinct_mesh.get_local_rank()) % 2**64

    dist_context.logger.info(f"Set seed {seed}")

    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
    random.seed(seed)
    np.random.seed(seed)

    # Set DTensor seeding if distributed
    if dist_context.mesh_params.is_distributed:
        mesh_regular = dist_context.mesh_for(REGULAR_DOMAIN)
        duplicate_seed_mesh_dim = tuple(
            name
            for name
            in cast(list[str], mesh_regular.mesh_dim_names)
            if name != distinct_seed_mesh_dim
        )
        duplicate_seed_mesh = mesh_regular[duplicate_seed_mesh_dim] if len(duplicate_seed_mesh_dim) != 0 else None

        if duplicate_seed_mesh and duplicate_seed_mesh.get_coordinate() is not None:
            torch.distributed.tensor._random.manual_seed(seed, duplicate_seed_mesh)  # noqa: SLF001
