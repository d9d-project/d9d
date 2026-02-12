from collections.abc import Sequence
from typing import TypeVar, cast

import torch
import torch.utils._pytree as pytree  # noqa: PLC2701

from d9d.core.types import PyTree

from .spec import ShardingSpec, ShardingSpecLeaf, SpecReplicate, SpecShard

TLeaf = TypeVar("TLeaf")
TSameTree = TypeVar("TSameTree", bound=PyTree)


def _unshard_list(group: Sequence[list[TLeaf] | TLeaf], spec: SpecShard) -> list[TLeaf]:
    if spec.dim != 0:
        raise ValueError(f"Lists can only be unsharded on dim 0, got {spec.dim}")

    if spec.do_stack:
        return cast(list[TLeaf], list(group))

    merged_list: list[TLeaf] = []
    for x in group:
        merged_list.extend(cast(list[TLeaf], x))
    return merged_list


def _unshard_tensor(group: list[torch.Tensor], spec: SpecShard) -> torch.Tensor:
    if spec.do_stack:
        return torch.stack(group, dim=spec.dim)

    return torch.cat(group, dim=spec.dim)


def _unshard_leaf_from_group(group: Sequence[TLeaf], spec: ShardingSpecLeaf) -> TLeaf:
    """Helper to merge a group of items from different ranks into one."""
    if isinstance(spec, SpecReplicate):
        return group[0]

    if not isinstance(spec, SpecShard):
        raise TypeError(f"Unknown sharding spec object type: {type(spec)}")

    first_item = group[0]

    if isinstance(first_item, torch.Tensor):
        return cast(TLeaf, _unshard_tensor(cast(list[torch.Tensor], group), spec))
    elif spec.do_stack or isinstance(first_item, list):
        return cast(TLeaf, _unshard_list(group, spec))
    else:
        raise TypeError(f"Expected Tensor or list instances, got {type(group[0])}")


def unshard_tree(sharded_trees: Sequence[TSameTree], sharding_spec: ShardingSpec) -> TSameTree:
    """
    Combines a sequence of PyTrees (one per rank) into a single global PyTree.

    This is the inverse of ``shard_tree``. It iterates over the provided trees,
    gathering corresponding leaves from each rank.

    *   If the spec for a leaf is ``SpecShard(dim)``, the tensors from all ranks are
        concatenated (or stacked if ``do_stack=True``) along that dimension.
    *   If the spec is ``SpecReplicate``, it assumes the data is replicated
        and takes the item from the first rank.

    Args:
        sharded_trees: A sequence (list or tuple) of PyTrees. There must be
            one tree for each shard rank, and they must all share the same
            structure as ``sharding_spec``.
        sharding_spec: A structure matching the input trees containing
            ``SpecShard`` or ``SpecReplicate`` objects.

    Returns:
        A single PyTree where distinct shards have been merged into full tensors.

    Raises:
        ValueError: If ``sharded_trees`` is empty, or if unit structures do
            not match the spec.
    """
    if not sharded_trees:
        raise ValueError("sharded_trees sequence cannot be empty")

    flat_spec, spec_struct = pytree.tree_flatten(sharding_spec)

    flat_shards_per_rank = []
    for i, tree in enumerate(sharded_trees):
        try:
            leaves = spec_struct.flatten_up_to(tree)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Structure mismatch at shard {i}: tree does not match sharding spec structure") from e

        flat_shards_per_rank.append(leaves)

    grouped_leaves = list(zip(*flat_shards_per_rank, strict=True))

    reconstructed_leaves = [
        _unshard_leaf_from_group(group, spec) for group, spec in zip(grouped_leaves, flat_spec, strict=True)
    ]

    return spec_struct.unflatten(reconstructed_leaves)
