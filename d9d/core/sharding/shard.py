from collections.abc import Sequence
from typing import TypeVar

import torch
import torch.utils._pytree as pytree  # noqa: PLC2701
from torch.distributed.tensor import Shard

from d9d.core.sharding import ShardingSpec
from d9d.core.types import PyTree

TLeaf = TypeVar("TLeaf")
TSameTree = TypeVar("TSameTree", bound=PyTree)


def _shard_leaf_to_list(
        item: TLeaf,
        spec: Shard | None,
        num_shards: int,
        enforce_even_split: bool
) -> Sequence[TLeaf | torch.Tensor]:
    """Helper to split an item into a list of items for each rank."""
    if spec is None:
        # Replicated: strict copy reference for all shards
        return [item] * num_shards

    if not isinstance(spec, Shard):
        raise TypeError(f"Unknown sharding spec object type: {type(spec)}")
    if not isinstance(item, torch.Tensor):
        raise TypeError(f"Sharding spec found a Shard object, but the item was not a Tensor (got {type(item)})")
    if item.ndim == 0:
        raise ValueError("Found a 0-dim Tensor for sharding")

    if enforce_even_split and item.shape[spec.dim] % num_shards != 0:
        raise ValueError(
            f"Tried to shard a tensor with shape {item.shape} on dim {spec.dim} "
            f"to {num_shards} shards, but the dimension is not perfectly divisible."
        )

    return torch.tensor_split(item, sections=num_shards, dim=spec.dim)


def shard_tree(
        tree: TSameTree,
        sharding_spec: ShardingSpec,
        num_shards: int,
        enforce_even_split: bool
) -> tuple[TSameTree, ...]:
    """
    Shards a PyTree into a tuple of PyTrees, one for each shard rank.

    This takes a single global data structure and splits it into ``num_shards``
    structures.

    If a spec leaf is a ``Shard(dim)``, the tensor is split along that dimension,
    and the ``i``-th slice goes to the ``i``-th output tree.
    If a spec leaf is ``None``, the item is replicated (reference copy) to all
    output trees.

    Args:
        tree: The structure containing tensors to be sharded.
        sharding_spec: A structure matching 'tree' containing ``Shard`` objects
            or None.
        num_shards: The total number of shards to split the tensors into.
        enforce_even_split: If True, raises a ValueError if a tensor's dimension
            size is not perfectly divisible by ``num_shards``.

    Returns:
        A tuple of length ``num_shards``. Each element is a PyTree matching
        the structure of the input ``tree``, containing the local data for
        that specific rank.

    Raises:
        ValueError: If tree structures do not match, or valid sharding conditions
            are not met.
    """
    # 1. Flatten inputs to process leaves linearly
    flat_tree, tree_struct = pytree.tree_flatten(tree)
    flat_spec, _ = pytree.tree_flatten(sharding_spec)

    # 2. Check structure compatibility (lengths must match)
    if len(flat_tree) != len(flat_spec):
        raise ValueError(
            f"Tree structure mismatch: tree has {len(flat_tree)} leaves "
            f"but spec has {len(flat_spec)} leaves."
        )

    # 3. Process leaves: List[Any] -> List[Sequence[Any]]
    # Each leaf becomes a sequence of length 'num_shards'
    sharded_leaves_per_node = [
        _shard_leaf_to_list(item, spec, num_shards, enforce_even_split)
        for item, spec in zip(flat_tree, flat_spec, strict=True)
    ]

    # 4. Transpose: List[Sequence[Any]] -> Sequence[List[Any]]
    # From "List of N-tuples" to "N-tuple of Lists"
    # rank_leaves[i] contains all leaves for rank i
    rank_leaves = list(zip(*sharded_leaves_per_node, strict=True))

    # 5. Reconstruct N trees
    return tuple(tree_struct.unflatten(leaves) for leaves in rank_leaves)
