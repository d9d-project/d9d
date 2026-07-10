import dataclasses
from collections.abc import Callable
from typing import Any

import optree
import optree.dataclasses

_NAMESPACE = "d9d"

IsLeaf = Callable[[Any], bool]


class PyTreeFlattener:
    """Flattens pytrees into their leaves, descending into dataclasses as internal nodes."""

    def __init__(self):
        """Constructs a PyTreeFlattener."""
        self._registered: set[type] = set()
        self._is_dataclass_memo: dict[type, bool] = {}

    def flatten(self, tree: Any, is_leaf: IsLeaf | None = None) -> tuple[list[Any], optree.PyTreeSpec]:
        """Flattens ``tree``, lazily registering any dataclass types discovered as leaves.

        Args:
            tree: The nested structure to flatten.
            is_leaf: Optional predicate; when it returns ``True`` for a node, that node is treated as
                a leaf and not traversed further (even if it is a dataclass or container).

        Returns:
            A tuple of the leaf list and the ``optree`` structure specification.
        """
        while True:
            leaves, treespec = optree.tree_flatten(tree, is_leaf=is_leaf, namespace=_NAMESPACE)
            if self._register_unknown_nodes(leaves, is_leaf):
                return leaves, treespec

    def flatten_with_path(
        self, tree: Any, is_leaf: IsLeaf | None = None
    ) -> tuple[list[tuple[Any, ...]], list[Any], optree.PyTreeSpec]:
        """Flattens ``tree`` alongside the path to each leaf, lazily registering dataclass types.

        Args:
            tree: The nested structure to flatten.
            is_leaf: Optional predicate; when it returns ``True`` for a node, that node is treated as
                a leaf and not traversed further (even if it is a dataclass or container).

        Returns:
            A tuple of the per-leaf paths, the leaf list, and the ``optree`` structure specification.
        """
        while True:
            paths, leaves, treespec = optree.tree_flatten_with_path(tree, is_leaf=is_leaf, namespace=_NAMESPACE)
            if self._register_unknown_nodes(leaves, is_leaf):
                return paths, leaves, treespec

    def _register_unknown_nodes(self, leaves: list[Any], is_leaf: IsLeaf | None) -> bool:
        """Registers any not-yet-registered dataclass types among ``leaves``.

        Args:
            leaves: The leaves produced by a flatten pass.
            is_leaf: The predicate passed to the flatten call, if any.

        Returns:
            ``True`` if all leaves were already fully expanded (nothing to register), else ``False``.
        """
        unregistered = {
            type(leaf)
            for leaf in leaves
            if type(leaf) not in self._registered
            and self._is_dataclass_type(type(leaf))
            and not (is_leaf is not None and is_leaf(leaf))
        }
        if not unregistered:
            return True

        for cls in unregistered:
            self._register(cls)
        return False

    def _is_dataclass_type(self, tp: type) -> bool:
        cached = self._is_dataclass_memo.get(tp)
        if cached is None:
            cached = dataclasses.is_dataclass(tp)
            self._is_dataclass_memo[tp] = cached
        return cached

    def _register(self, cls: type) -> None:
        if cls in self._registered:
            return
        optree.dataclasses.register_node(cls, namespace=_NAMESPACE)
        self._registered.add(cls)
