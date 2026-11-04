import pytest
from d9d.rng import SeedScope, fork_rng


@pytest.fixture(autouse=True)
def _seed_fork():
    # EventBus.trigger() forks a child of the innermost fork, which the loop opens around every call.
    with fork_rng(SeedScope.root(0)):
        yield
