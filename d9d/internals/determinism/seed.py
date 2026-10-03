import os
import random
from typing import cast

import torch
import torch.distributed.tensor

from d9d.core.dist_context import REGULAR_DOMAIN, DistributedContext


def set_seeds(
    dist_context: DistributedContext,
    seed: int,
    distinct_seed_mesh_dim: str = "pp",
) -> None:
    """Sets the random seeds of Python, NumPy and PyTorch.

    The seed is the base seed plus the rank in ``distinct_seed_mesh_dim``, e.g. the pipeline-parallel rank.
    So different pipeline stages get different random states, while ranks along other dimensions share one.
    In a distributed setup, the function also seeds the ``DTensor`` random generator over the other dimensions.

    Args:
        dist_context: The distributed context.
        seed: The base random seed.
        distinct_seed_mesh_dim: The name of the mesh dimension along which seeds differ, e.g. ``"pp"``. Ranks
            along other dimensions share the seed.
    """
    if dist_context.mesh_params.is_distributed:
        distinct_mesh = dist_context.mesh_for(REGULAR_DOMAIN)[distinct_seed_mesh_dim]
        seed = (seed + distinct_mesh.get_local_rank()) % 2**64

    dist_context.logger.info(f"Set seed {seed}")

    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
    random.seed(seed)

    try:
        import numpy as np  # noqa: PLC0415

        np.random.seed(seed)
    except ImportError:
        pass

    if dist_context.mesh_params.is_distributed:
        mesh_regular = dist_context.mesh_for(REGULAR_DOMAIN)
        duplicate_seed_mesh_dim = tuple(
            name for name in cast(list[str], mesh_regular.mesh_dim_names) if name != distinct_seed_mesh_dim
        )
        duplicate_seed_mesh = mesh_regular[duplicate_seed_mesh_dim] if len(duplicate_seed_mesh_dim) != 0 else None

        if duplicate_seed_mesh and duplicate_seed_mesh.get_coordinate() is not None:
            # torch can seed the DTensor random generator for a given mesh only through a private function.
            torch.distributed.tensor._random.manual_seed(seed, duplicate_seed_mesh)  # noqa: SLF001 - no public API
