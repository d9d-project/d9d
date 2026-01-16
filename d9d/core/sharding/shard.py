from typing import Any

import torch
import torch.utils._pytree as pytree
from torch.distributed.tensor import Shard

from d9d.core.sharding import ShardingSpec


def _shard_leaf(item: Any, spec: Any, num_shards: int, enforce_even_split: bool) -> Any:
    """
    Helper function to shard a single leaf item.
    """
    if spec is None:
        return item

    if not isinstance(spec, Shard):
        raise ValueError(f'Unknown sharding spec object type: {type(spec)}')

    if not isinstance(item, torch.Tensor):
        raise ValueError(f"Sharding spec found a Shard object, but the item was not a Tensor (got {type(item)})")

    if item.ndim == 0:
        raise ValueError('Found a 0-dim Tensor for sharding')

    if enforce_even_split and item.shape[spec.dim] % num_shards != 0:
        raise ValueError(f'Tried to shard a tensor with shape {item.shape} on dim {spec.dim} to {num_shards} shards')

    return torch.tensor_split(item, sections=num_shards, dim=spec.dim)


def shard_tree(
        tree: Any,
        sharding_spec: ShardingSpec,
        num_shards: int,
        enforce_even_split: bool
) -> Any:
    """
    Shards a PyTree structure by slicing tensors according to the provided specification.

    This function iterates over a data tree and a corresponding sharding specification tree.
    If a spec leaf is a ``Shard`` object, the corresponding tensor is split along
    the specified dimension.
    If the spec leaf is None, the original item is returned as-is.

    Args:
        tree: The structure containing tensors to be sharded.
        sharding_spec: A structure matching 'tree' containing ``Shard`` objects or None,
            typically created via ``d9d.core.sharding.shard_on_dim``.
        num_shards: The total number of shards to split the tensors into.
        enforce_even_split: If True, raises a ValueError if a tensor's dimension size
            is not perfectly divisible by ``num_shards``. This guarantees that all
            resulting shards have identical shapes. If False, the function uses
            ``torch.tensor_split`` behavior, where the first few shards may contain
            one element more than the others.

    Returns:
        A new PyTree with the same structure as input, where tensors marked for
        sharding have been sliced.

    Raises:
        ValueError: If tree structures do not match, if a Shard spec is applied
            to a non-Tensor or a 0-dim Tensor, or if strict divisibility is
            enforced but not met.
    """
    return pytree.tree_map(
        lambda t, s: _shard_leaf(t, s, num_shards=num_shards, enforce_even_split=enforce_even_split),
        tree,
        sharding_spec
    )
