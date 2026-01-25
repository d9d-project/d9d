from collections.abc import Sequence
from typing import TypeVar, cast

import torch
import torch.utils._pytree as pytree  # noqa: PLC2701
from torch.distributed.tensor import Shard

from d9d.core.sharding import ShardingSpec
from d9d.core.types import PyTree

TLeaf = TypeVar("TLeaf")
TSameTree = TypeVar("TSameTree", bound=PyTree)


def _shard_list(
        item: list[TLeaf],
        spec: Shard,
        num_shards: int,
        enforce_even_split: bool
) -> Sequence[list[TLeaf]]:
    if spec.dim != 0:
        raise ValueError(f"Lists can only be sharded on dim 0, got {spec.dim}")

    if enforce_even_split and len(item) % num_shards != 0:
        raise ValueError(
            f"Tried to shard a list with length {len(item)} "
            f"to {num_shards} shards, but the length is not perfectly divisible."
        )

    shard_size, shard_extra = divmod(len(item), num_shards)
    return [
        item[
            shard_id * shard_size + min(shard_id, shard_extra):
            (shard_id + 1) * shard_size + min(shard_id + 1, shard_extra)
        ]
        for shard_id in range(num_shards)
    ]


def _shard_tensor(
        item: torch.Tensor,
        spec: Shard,
        num_shards: int,
        enforce_even_split: bool
) -> Sequence[torch.Tensor]:
    if item.ndim == 0:
        raise ValueError("Found a 0-dim Tensor for sharding")

    if enforce_even_split and item.shape[spec.dim] % num_shards != 0:
        raise ValueError(
            f"Tried to shard a tensor with shape {item.shape} on dim {spec.dim} "
            f"to {num_shards} shards, but the dimension is not perfectly divisible."
        )

    return torch.tensor_split(item, sections=num_shards, dim=spec.dim)


def _shard_leaf_to_list(
        item: TLeaf,
        spec: Shard | None,
        num_shards: int,
        enforce_even_split: bool
) -> Sequence[TLeaf]:
    """Helper to split an item into a list of items for each rank."""
    if spec is None:
        # Replicated: strict copy reference for all shards
        return [item] * num_shards

    if not isinstance(spec, Shard):
        raise TypeError(f"Unknown sharding spec object type: {type(spec)}")

    if isinstance(item, torch.Tensor):
        return cast(Sequence[TLeaf], _shard_tensor(
            item=item,
            num_shards=num_shards,
            enforce_even_split=enforce_even_split,
            spec=spec
        ))
    elif isinstance(item, list):
        return cast(Sequence[TLeaf], _shard_list(
            item=item,
            num_shards=num_shards,
            enforce_even_split=enforce_even_split,
            spec=spec
        ))
    else:
        raise TypeError(
            f"Sharding spec found a Shard object, but the item was not a Tensor and not a list (got {type(item)})"
        )


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
    flat_spec, spec_struct = pytree.tree_flatten(sharding_spec)

    try:
        flat_tree = spec_struct.flatten_up_to(tree)
    except (ValueError, TypeError) as e:
        raise ValueError("Tree structure does not match sharding spec") from e

    sharded_leaves_per_node = [
        _shard_leaf_to_list(item, spec, num_shards, enforce_even_split)
        for item, spec in zip(flat_tree, flat_spec, strict=True)
    ]

    rank_leaves = list(zip(*sharded_leaves_per_node, strict=True))

    # 5. Reconstruct N trees
    return tuple(spec_struct.unflatten(leaves) for leaves in rank_leaves)
