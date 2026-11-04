import pytest
from d9d.rng import SeedScope


@pytest.mark.local
def test_seed_scope_is_stable():
    # The seeds must not change between processes, Python versions and releases of d9d.
    assert SeedScope.root(42).seed == 11856747883583595420
    assert SeedScope.root(42).child("pass", 3, 0, 1, 0).seed == 1848848409464668940


@pytest.mark.local
@pytest.mark.parametrize(
    ("left", "right"),
    [
        (("a",), ("b",)),
        (("1",), (1,)),
        (("ab", "c"), ("a", "bc")),
        ((1, 2), (2, 1)),
        ((), ("a",)),
    ],
)
def test_seed_scope_children_differ(left, right):
    root = SeedScope.root(0)
    assert root.child(*left).seed != root.child(*right).seed


@pytest.mark.local
def test_seed_scope_root_accepts_any_integer():
    seeds = {SeedScope.root(base_seed).seed for base_seed in (-1, 0, 1, 2**70)}
    assert len(seeds) == 4


@pytest.mark.local
@pytest.mark.parametrize("seed", [-1, 2**64])
def test_seed_scope_rejects_out_of_range(seed):
    with pytest.raises(ValueError, match="must be in"):
        SeedScope(seed=seed)
