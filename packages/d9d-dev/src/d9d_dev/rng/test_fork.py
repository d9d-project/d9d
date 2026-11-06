import contextlib

import pytest
import torch
from d9d.rng import SeedScope, fork_rng, fork_rng_child


@pytest.fixture(params=["cpu", "cuda"])
def device(request) -> str:
    return request.param


@pytest.mark.local
def test_fork_rng_seeds_from_scope(device):
    scope = SeedScope.root(0).child("a")

    with fork_rng(scope):
        first = torch.rand(4, device=device)
    with fork_rng(scope):
        second = torch.rand(4, device=device)
    with fork_rng(SeedScope.root(0).child("b")):
        other = torch.rand(4, device=device)

    assert torch.equal(first, second)
    assert not torch.equal(first, other)


@pytest.mark.local
@pytest.mark.parametrize("fail", [False, True])
def test_fork_rng_restores_outer_state(device, fail):
    torch.manual_seed(0)
    expected = torch.rand(4, device=device)

    torch.manual_seed(0)
    exit_check = pytest.raises(RuntimeError, match="boom") if fail else contextlib.nullcontext()
    with exit_check, fork_rng(SeedScope.root(0)):
        torch.rand(4, device=device)
        # A nested fork must not shift the numbers of the outer fork either.
        with fork_rng(SeedScope.root(1)):
            torch.rand(4, device=device)
        if fail:
            raise RuntimeError("boom")
    actual = torch.rand(4, device=device)

    assert torch.equal(expected, actual)


@pytest.mark.local
@pytest.mark.parametrize("fail", [False, True])
def test_reentered_fork_continues_its_numbers(device, fail):
    with fork_rng(SeedScope.root(0)):
        expected = torch.cat([torch.rand(4, device=device), torch.rand(4, device=device)])

    fork = fork_rng(SeedScope.root(0))
    exit_check = pytest.raises(RuntimeError, match="boom") if fail else contextlib.nullcontext()
    with exit_check, fork:
        first = torch.rand(4, device=device)
        if fail:
            raise RuntimeError("boom")
    with fork:
        second = torch.rand(4, device=device)

    assert torch.equal(torch.cat([first, second]), expected)
    with fork, pytest.raises(RuntimeError, match="already open"), fork:
        pass


@pytest.mark.local
@pytest.mark.parametrize("fail", [False, True])
def test_fork_rng_child_follows_innermost_fork(fail):
    def draw_child() -> torch.Tensor:
        with fork_rng_child("x"):
            return torch.rand(4)

    def draw_expected(scope: SeedScope) -> torch.Tensor:
        with fork_rng(scope.child("x")):
            return torch.rand(4)

    outer, inner = SeedScope.root(0), SeedScope.root(1)

    exit_check = pytest.raises(RuntimeError, match="boom") if fail else contextlib.nullcontext()
    with exit_check, fork_rng(outer):
        with fork_rng(inner):
            assert torch.equal(draw_child(), draw_expected(inner))
        assert torch.equal(draw_child(), draw_expected(outer))
        if fail:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="outside of fork_rng"):
        draw_child()


@pytest.mark.local
def test_fork_rng_child_forks_from_current_seed_scope():
    scope = SeedScope.root(0)

    with fork_rng(scope), fork_rng_child("a", 1):
        actual = torch.rand(4)
    with fork_rng(scope.child("a", 1)):
        expected = torch.rand(4)

    assert torch.equal(actual, expected)


@pytest.mark.local
def test_fork_rng_child_outside_fork_raises():
    with pytest.raises(RuntimeError, match="outside of fork_rng"):
        fork_rng_child("a")
