from collections.abc import Callable
from typing import Any, TypeVar, cast

import optree

from d9d.core.types import PyTree

from .flatten import IsLeaf, PyTreeFlattener

TLeaf = TypeVar("TLeaf")
TMapped = TypeVar("TMapped")
TTree = TypeVar("TTree", bound=PyTree)

PyTreeSpec = optree.PyTreeSpec


_flattener = PyTreeFlattener()


def tree_flatten(tree: PyTree[TLeaf], is_leaf: IsLeaf | None = None) -> tuple[list[TLeaf], PyTreeSpec]:
    """Flattens a pytree into its leaves and a structure specification.

    Args:
        tree: The nested structure to flatten.
        is_leaf: Optional predicate; when it returns ``True`` for a node, that node is kept as a leaf
            and not traversed further.

    Returns:
        A tuple of the leaf list and a ``PyTreeSpec`` that can rebuild the structure via
        `tree_unflatten`.
    """
    return _flattener.flatten(tree, is_leaf)


def tree_unflatten(treespec: PyTreeSpec, leaves: list[TLeaf]) -> PyTree[TLeaf]:
    """Reconstructs a pytree from leaves and a structure specification.

    Args:
        treespec: A specification produced by `tree_flatten`.
        leaves: The leaves to place into the structure, in flatten order.

    Returns:
        The reconstructed nested structure.
    """
    return optree.tree_unflatten(treespec, leaves)


def tree_leaves(tree: PyTree[TLeaf], is_leaf: IsLeaf | None = None) -> list[TLeaf]:
    """Returns the leaves of a pytree in deterministic (sorted-key) order.

    Args:
        tree: The nested structure to flatten.
        is_leaf: Optional predicate; when it returns ``True`` for a node, that node is kept as a leaf
            and not traversed further.

    Returns:
        The list of leaves.
    """
    return _flattener.flatten(tree, is_leaf)[0]


def tree_map(func: Callable[[TLeaf], TMapped], tree: PyTree[TLeaf]) -> PyTree[TMapped]:
    """Applies ``func`` to every leaf of a pytree, returning a structurally-identical tree.

    Args:
        func: The function to apply to each leaf.
        tree: The nested structure to map over.

    Returns:
        A new tree with ``func`` applied to each leaf.
    """
    leaves, treespec = _flattener.flatten(tree)
    return optree.tree_unflatten(treespec, [func(leaf) for leaf in leaves])


def tree_map_only(
    filters: type | tuple[type, ...],
    func: Callable[[Any], Any],
    tree: TTree,
) -> TTree:
    """Applies ``func`` only to leaves that are instances of ``type_or_types``.

    Leaves of any other type are returned unchanged. This is the common case for tensor
    operations over trees that also carry non-tensor bookkeeping (e.g. moving only tensors to a
    device while leaving strings and ints alone). The returned tree preserves the structure and
    leaf types of the input.

    Args:
        filters: The leaf type(s) that ``func`` should be applied to.
        func: The function to apply to matching leaves.
        tree: The nested structure to map over.

    Returns:
        A new tree with ``func`` applied to matching leaves only.
    """
    leaves, treespec = _flattener.flatten(tree)
    mapped = [func(leaf) if isinstance(leaf, filters) else leaf for leaf in leaves]
    return cast(TTree, optree.tree_unflatten(treespec, mapped))


def tree_leaves_with_path(tree: PyTree[TLeaf], is_leaf: IsLeaf | None = None) -> list[tuple[tuple[Any, ...], TLeaf]]:
    """Returns ``(path, leaf)`` pairs for every leaf of a pytree.

    Each path is a tuple of keys and indices reaching the leaf from the root: ``str`` for dict keys
    and dataclass field names, ``int`` for sequence indices.

    Args:
        tree: The nested structure to flatten.
        is_leaf: Optional predicate; when it returns ``True`` for a node, that node is kept as a leaf
            and not traversed further.

    Returns:
        A list of ``(path, leaf)`` tuples in deterministic (sorted-key) order.
    """
    paths, leaves, _ = _flattener.flatten_with_path(cast(Any, tree), is_leaf)
    return list(zip(paths, leaves, strict=True))
