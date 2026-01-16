from typing import Any, List
import torch
import torch.utils._pytree as pytree
from torch.distributed.tensor import Shard

from d9d.core.sharding import ShardingSpec


def _unshard_leaf(item: Any, spec: Any) -> Any:
    """Concatenates a sequence of tensors if the spec is a Shard."""

    if spec is None:
        return item

    if not isinstance(spec, Shard):
        raise ValueError(f'Unknown sharding spec object type: {type(spec)}')

    if not isinstance(item, (tuple, list)):
        raise ValueError(
            f"Expected a sequence (tuple/list) of tensors to unshard for spec {spec}, "
            f"but got {type(item)}. This often indicates a mismatch between "
            "the sharding spec structure and the sharded data structure."
        )

    if len(item) == 0:
        raise ValueError(f"Found empty sequence of shards for spec {spec}")

    return torch.cat(item, dim=spec.dim)


def unshard_tree(
        tree: Any,
        sharding_spec: ShardingSpec
) -> Any:
    """
    Reconstructs a PyTree of tensors by concatenating shards.

    This function uses structural definitions from ``sharding_spec`` to traverse
    the ``tree``. It descends into the ``tree`` only as deep as the
    ``sharding_spec`` goes. If the spec has aleaf leaf (either ``Shard`` or ``None``),
    the corresponding substructure in ``tree`` is treated as a single unit
    (e.g., a tuple of shards) and passed to the concatenation logic.

    Args:
        tree: The structure containing sequences of tensor shards.
        sharding_spec: A structure matching 'tree' containing ``Shard`` objects or None.

    Returns:
        A new PyTree where sequences of shards have been concatenated back into
        single tensors.

    Raises:
        ValueError: If tree structures do not match, or if a ``Shard`` spec is
            applied to a non-sequence item.
    """
    spec_leaves, spec_struct = pytree.tree_flatten(sharding_spec)

    try:
        grouped_shards = spec_struct.flatten_up_to(tree)
    except ValueError as e:
        raise ValueError(
            f"Structure of 'tree' does not match structure of 'sharding_spec'. "
            f"Details: {e}"
        ) from e

    reconstructed_leaves = [
        _unshard_leaf(group, spec)
        for group, spec in zip(grouped_shards, spec_leaves)
    ]

    return spec_struct.unflatten(reconstructed_leaves)
