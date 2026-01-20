from typing import Any

import torch
import torch.utils._pytree as pytree  # noqa: PLC2701
from torch.distributed.tensor import Shard

ShardingSpec = Any


def _tree_item_to_shard(item: Any, shard_on_dim: int) -> Shard | None:
    if not isinstance(item, torch.Tensor):
        return None
    if item.ndim == 0:
        return None
    if item.ndim <= shard_on_dim:
        raise ValueError(f"Cannot shard {item.ndim}-dimensional tensor on dim {shard_on_dim}")
    return Shard(shard_on_dim)


def shard_spec_on_dim(tree: Any, dim: int) -> ShardingSpec:
    """
    Creates a sharding specification to split all tensors in the tree on a specific dimension.

    Iterates over the input tree. If a leaf is a Tensor with enough dimensions,
    it is mapped to a Shard object. Non-tensors and 0-dim tensors are mapped to None.

    Args:
        tree: The input PyTree structure.
        dim: The dimension index to shard eligible tensors on.

    Returns:
        A new PyTree matching the input structure, containing Shard objects or None.

    Raises:
        ValueError: If a tensor exists in the tree with rank less than or equal to 'dim'.
    """

    return pytree.tree_map(lambda x: _tree_item_to_shard(x, dim), tree)


def shard_spec_nothing(tree: Any) -> ShardingSpec:
    """
    Creates a sharding specification where no sharding is performed.

    This effectively clones the tree structure but replaces every leaf with None.

    Args:
        tree: The input PyTree structure.

    Returns:
        A new PyTree matching the input structure, containing strictly None for all leaves.
    """

    return pytree.tree_map(lambda _: None, tree)
