import dataclasses
from typing import Any

import optree
import optree.dataclasses

_NAMESPACE = "d9d"


class PyTreeFlattener:
    """Flattens pytrees into their leaves, descending into dataclasses as internal nodes."""

    def __init__(self):
        """Constructs a PyTreeFlattener."""
        self._registered: set[type] = set()
        self._is_dataclass_memo: dict[type, bool] = {}

    def flatten(self, tree: Any) -> tuple[list[Any], optree.PyTreeSpec]:
        """Flattens ``tree``, lazily registering any dataclass types discovered as leaves.

        Args:
            tree: The nested structure to flatten.

        Returns:
            A tuple of the leaf list and the ``optree`` structure specification.
        """
        while True:
            leaves, treespec = optree.tree_flatten(tree, namespace=_NAMESPACE)
            if self._register_unknown_nodes(leaves):
                return leaves, treespec

    def flatten_with_path(self, tree: Any) -> tuple[list[tuple[Any, ...]], list[Any], optree.PyTreeSpec]:
        """Flattens ``tree`` alongside the path to each leaf, lazily registering dataclass types.

        Args:
            tree: The nested structure to flatten.

        Returns:
            A tuple of the per-leaf paths, the leaf list, and the ``optree`` structure specification.
        """
        while True:
            paths, leaves, treespec = optree.tree_flatten_with_path(tree, namespace=_NAMESPACE)
            if self._register_unknown_nodes(leaves):
                return paths, leaves, treespec

    def _register_unknown_nodes(self, leaves: list[Any]) -> bool:
        """Registers any not-yet-registered dataclass types among ``leaves``.

        Args:
            leaves: The leaves produced by a flatten pass.

        Returns:
            ``True`` if all leaves were already fully expanded (nothing to register), else ``False``.
        """
        unregistered = {
            type(leaf) for leaf in leaves if type(leaf) not in self._registered and self._is_dataclass_type(type(leaf))
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
