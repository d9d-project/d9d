from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.utils._pytree as pytree  # noqa: PLC2701
from torch.distributed.tensor import Shard

from d9d.core.sharding import ShardingSpec


def _unshard_leaf_from_group(
        group: Sequence[Any],
        spec: Any
) -> Any:
    """Helper to merge a group of items from different ranks into one."""
    if spec is None:
        # Replicated: All ranks should have the same item.
        # We simply return the first one.
        return group[0]

    if not isinstance(spec, Shard):
        raise TypeError(f"Unknown sharding spec object type: {type(spec)}")

    # Validation: Items must be tensors
    if not isinstance(group[0], torch.Tensor):
        raise TypeError(f"Expected Tensors for Shard spec, got {type(group[0])}")

    group = cast(list[torch.Tensor], group)  # if first element is tensor - skip runtime checks

    return torch.cat(group, dim=spec.dim)


def unshard_tree(
        sharded_trees: Sequence[Any],
        sharding_spec: ShardingSpec
) -> Any:
    """
    Combines a sequence of PyTrees (one per rank) into a single global PyTree.

    This is the inverse of ``shard_tree``. It iterates over the provided trees,
    gathering corresponding leaves from each rank.

    If the spec for a leaf is ``Shard(dim)``, the tensors from all ranks are
    concatenated along that dimension.
    If the spec is ``None``, it assumes the data is replicated and takes the
    item from the first rank.

    Args:
        sharded_trees: A sequence (list or tuple) of PyTrees. There must be
            one tree for each shard rank, and they must all share the same
            structure as ``sharding_spec``.
        sharding_spec: A structure matching the input trees containing
            ``Shard`` objects or None.

    Returns:
        A single PyTree where distinct shards have been merged into full tensors.

    Raises:
        ValueError: If ``sharded_trees`` is empty, or if unit structures do
            not match the spec.
    """
    if not sharded_trees:
        raise ValueError("sharded_trees sequence cannot be empty")

    # 1. Flatten the spec
    flat_spec, spec_struct = pytree.tree_flatten(sharding_spec)

    # 2. Flatten all input trees
    # flat_shards_per_rank: List[List[Any]] -> One list of leaves per rank
    flat_shards_per_rank = []
    for i, tree in enumerate(sharded_trees):
        leaves, _ = pytree.tree_flatten(tree)
        if len(leaves) != len(flat_spec):
            raise ValueError(
                f"Structure mismatch at rank {i}: tree has {len(leaves)} leaves "
                f"but spec has {len(flat_spec)}."
            )
        # Note: We assume 'struct' matches 'spec_struct' if lengths match
        # because pytree traversal is deterministic.
        flat_shards_per_rank.append(leaves)

    # 3. Transpose: Sequence[List[Any]] -> List[Sequence[Any]]
    # We want to group the i-th leaf from all ranks together
    # grouped_leaves[i] = (leaf_i_rank0, leaf_i_rank1, ...)
    grouped_leaves = list(zip(*flat_shards_per_rank, strict=True))

    # 4. Merge groups
    reconstructed_leaves = [
        _unshard_leaf_from_group(group, spec)
        for group, spec in zip(grouped_leaves, flat_spec, strict=True)
    ]

    # 5. Reconstruct single tree
    return spec_struct.unflatten(reconstructed_leaves)
