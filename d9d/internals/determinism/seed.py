import os
import random

import torch
import numpy as np

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

    mesh_regular = dist_context.mesh_for(REGULAR_DOMAIN)

    distinct_mesh = mesh_regular[distinct_seed_mesh_dim]
    seed = (seed + distinct_mesh.get_local_rank()) % 2**64

    dist_context.logger.info(f'Set seed {seed}')

    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
    random.seed(seed)
    np.random.seed(seed)

    duplicate_seed_mesh = [name for name in mesh_regular.mesh_dim_names if name != distinct_seed_mesh_dim]
    duplicate_seed_mesh = mesh_regular[duplicate_seed_mesh] if len(duplicate_seed_mesh) else None

    if duplicate_seed_mesh and duplicate_seed_mesh.get_coordinate() is not None:
        torch.distributed.tensor._random.manual_seed(seed, duplicate_seed_mesh)
