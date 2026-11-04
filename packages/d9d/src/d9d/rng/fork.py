from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar

import torch

from .seed_scope import SeedScope

_current_seed_scope: ContextVar[SeedScope | None] = ContextVar("current_seed_scope", default=None)


@contextmanager
def fork_rng(scope: SeedScope) -> Iterator[None]:
    """Seeds the global generators from a seed scope while the context is open.

    The global generators are the torch CPU generator and the CUDA generator of the current device.
    On exit, the global generators return to the states they had before the context.

    Args:
        scope: The seed scope to take the seed from.
    """
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    token = _current_seed_scope.set(scope)
    try:
        with torch.random.fork_rng(devices=devices):
            # torch.manual_seed() would also seed every other CUDA device, which costs about 50 us per call.
            torch.default_generator.manual_seed(scope.seed)
            if devices:
                torch.cuda.manual_seed(scope.seed)
            yield
    finally:
        _current_seed_scope.reset(token)


def fork_rng_child(*keys: str | int) -> AbstractContextManager[None]:
    """Forks the global generators from a child of the seed scope of the innermost open ``fork_rng()``.

    Args:
        *keys: The keys of the child.

    Returns:
        A context manager that holds the fork while it is open.

    Raises:
        RuntimeError: If no ``fork_rng()`` is open on this thread.
    """
    scope = _current_seed_scope.get()
    if scope is None:
        raise RuntimeError(f"fork_rng_child() with keys ({keys}) runs outside of fork_rng(). Open fork_rng() first.")
    return fork_rng(scope.child(*keys))
