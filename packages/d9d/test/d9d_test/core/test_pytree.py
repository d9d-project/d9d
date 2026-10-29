import dataclasses

import pytest
import torch
from d9d.core import pytree


@dataclasses.dataclass
class Inner:
    a: torch.Tensor
    tag: str


@dataclasses.dataclass
class Outer:
    inner: Inner
    b: torch.Tensor


@dataclasses.dataclass(frozen=True)
class FrozenPoint:
    x: torch.Tensor


@dataclasses.dataclass(slots=True)
class SlottedPair:
    y: torch.Tensor
    z: torch.Tensor


def _leaf_equal(a: object, b: object) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    return a == b


# (tree, expected leaves in deterministic traversal order). Covers plain containers, nested
# dataclasses, dataclasses nested inside containers, and frozen/slotted dataclasses.
_STRUCTURES = [
    pytest.param(
        {"a": [torch.tensor(1.0), torch.tensor(2.0)], "b": (torch.tensor(3.0),)},
        [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)],
        id="plain-containers",
    ),
    pytest.param(
        Outer(inner=Inner(a=torch.tensor(1.0), tag="x"), b=torch.tensor(2.0)),
        [torch.tensor(1.0), "x", torch.tensor(2.0)],
        id="nested-dataclass",
    ),
    pytest.param(
        {"k": Inner(a=torch.tensor(1.0), tag="y"), "t": torch.tensor(2.0)},
        [torch.tensor(1.0), "y", torch.tensor(2.0)],
        id="dataclass-inside-dict",
    ),
    pytest.param(
        [Inner(a=torch.tensor(0.0), tag="m0"), Inner(a=torch.tensor(1.0), tag="m1")],
        [torch.tensor(0.0), "m0", torch.tensor(1.0), "m1"],
        id="list-of-dataclasses",
    ),
    pytest.param(FrozenPoint(x=torch.tensor(1.0)), [torch.tensor(1.0)], id="frozen-dataclass"),
    pytest.param(
        SlottedPair(y=torch.tensor(1.0), z=torch.tensor(2.0)),
        [torch.tensor(1.0), torch.tensor(2.0)],
        id="slotted-dataclass",
    ),
]


@pytest.mark.local
@pytest.mark.parametrize(("tree", "expected_leaves"), _STRUCTURES)
def test_flatten_and_map_descend_into_structure(tree, expected_leaves):
    leaves, treespec = pytree.tree_flatten(tree)
    assert len(leaves) == len(expected_leaves)
    assert all(_leaf_equal(actual, expect) for actual, expect in zip(leaves, expected_leaves, strict=True))

    result = pytree.tree_map(lambda t: t + 10 if isinstance(t, torch.Tensor) else t, tree)
    mapped_leaves, mapped_spec = pytree.tree_flatten(result)
    # Structure, including dataclass types, is preserved by the map.
    assert mapped_spec == treespec
    expected_mapped = [x + 10 if isinstance(x, torch.Tensor) else x for x in expected_leaves]
    assert all(_leaf_equal(actual, expect) for actual, expect in zip(mapped_leaves, expected_mapped, strict=True))


@pytest.mark.local
@pytest.mark.parametrize(
    ("tree", "expected_paths"),
    [
        pytest.param({"m": [torch.tensor(1.0), torch.tensor(2.0)]}, [("m", 0), ("m", 1)], id="dict-and-sequence"),
        pytest.param(
            Outer(inner=Inner(a=torch.tensor(1.0), tag="x"), b=torch.tensor(2.0)),
            [("inner", "a"), ("inner", "tag"), ("b",)],
            id="dataclass-field-names",
        ),
    ],
)
def test_leaves_with_path(tree, expected_paths):
    paths = [path for path, _ in pytree.tree_leaves_with_path(tree)]
    assert paths == expected_paths


@pytest.mark.local
def test_tree_map_only_filters_by_type():
    tree = {"k": Inner(a=torch.tensor(1.0, requires_grad=True), tag="y")}
    result = pytree.tree_map_only(torch.Tensor, lambda t: t.detach(), tree)
    assert not result["k"].a.requires_grad
    assert result["k"].tag == "y"


@pytest.mark.local
def test_tree_leaves_dict_key_sort_determinism():
    assert pytree.tree_leaves({"b": 2, "a": 1, "c": 3}) == [1, 2, 3]
    assert pytree.tree_leaves({"c": 3, "a": 1, "b": 2}) == [1, 2, 3]


@pytest.mark.local
def test_flatten_unflatten_roundtrip():
    tree = {
        "a": [torch.tensor(1.0), Inner(a=torch.tensor(2.0), tag="z")],
        "b": Outer(inner=Inner(a=torch.tensor(3.0), tag="q"), b=torch.tensor(4.0)),
    }
    leaves, treespec = pytree.tree_flatten(tree)
    rebuilt = pytree.tree_unflatten(treespec, leaves)
    assert rebuilt == tree


@dataclasses.dataclass(frozen=True)
class OpaqueLeaf:
    value: int


@pytest.mark.local
def test_is_leaf_keeps_matching_node_as_leaf():
    tree = {"a": OpaqueLeaf(1), "b": [OpaqueLeaf(2), OpaqueLeaf(3)]}
    leaves = pytree.tree_leaves(tree, is_leaf=lambda x: isinstance(x, OpaqueLeaf))
    assert leaves == [OpaqueLeaf(1), OpaqueLeaf(2), OpaqueLeaf(3)]


@pytest.mark.local
def test_is_leaf_roundtrip():
    tree = {"a": OpaqueLeaf(1), "b": OpaqueLeaf(2)}
    leaves, treespec = pytree.tree_flatten(tree, is_leaf=lambda x: isinstance(x, OpaqueLeaf))
    assert pytree.tree_unflatten(treespec, leaves) == tree


@pytest.mark.local
def test_is_leaf_does_not_globally_register_the_stopped_type():
    # A dataclass kept as a leaf via is_leaf must NOT be registered: a later plain flatten must
    # still descend into it, proving the registry was not mutated as a side effect.
    @dataclasses.dataclass
    class Boundary:
        a: torch.Tensor
        b: torch.Tensor

    stopped = pytree.tree_leaves(
        {"k": Boundary(torch.tensor(1.0), torch.tensor(2.0))}, is_leaf=lambda x: isinstance(x, Boundary)
    )
    assert len(stopped) == 1
    assert isinstance(stopped[0], Boundary)

    # Without is_leaf, the same type is descended into (registered lazily now, not before).
    descended = pytree.tree_leaves({"k": Boundary(torch.tensor(1.0), torch.tensor(2.0))})
    assert len(descended) == 2
    assert all(isinstance(leaf, torch.Tensor) for leaf in descended)


@pytest.mark.local
def test_lazily_discovered_nested_new_dataclass_type():
    # A dataclass type first seen only deep inside a container of an already-seen dataclass.
    @dataclasses.dataclass
    class DeepLeaf:
        v: torch.Tensor

    @dataclasses.dataclass
    class Wrapper:
        items: list

    tree = Wrapper(items=[{"nested": DeepLeaf(v=torch.tensor(5.0))}])
    result = pytree.tree_map(lambda t: t + 1 if isinstance(t, torch.Tensor) else t, tree)
    assert torch.equal(result.items[0]["nested"].v, torch.tensor(6.0))
