from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from types import TracebackType

import torch

from .seed_scope import SeedScope

_current_seed_scope: ContextVar[SeedScope | None] = ContextVar("current_seed_scope", default=None)


class _Fork(AbstractContextManager[None]):
    """A fork of the global generators that the code can enter several times.

    The first entry seeds the global generators from the scope. Each later entry continues the numbers where the
    previous entry stopped. On exit, the global generators return to the states they had before the entry.
    """

    def __init__(self, scope: SeedScope):
        self._scope = scope
        self._generators = [torch.default_generator]
        if torch.cuda.is_available():
            # Only the current device: torch.manual_seed() would seed every CUDA device, at about 50 us per call.
            device = torch.cuda.current_device()
            self._generators.append(torch.cuda.default_generators[device])
        self._own_states: list[torch.Tensor] | None = None
        self._outer_states: list[torch.Tensor] = []
        self._token: Token[SeedScope | None] | None = None

    def __enter__(self) -> None:
        if self._token is not None:
            raise RuntimeError("The fork is already open. Close it before you enter it again.")
        self._outer_states = [generator.get_state() for generator in self._generators]
        self._token = _current_seed_scope.set(self._scope)
        if self._own_states is None:
            for generator in self._generators:
                generator.manual_seed(self._scope.seed)
        else:
            for generator, state in zip(self._generators, self._own_states, strict=True):
                generator.set_state(state)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._token is None:
            raise RuntimeError("The fork is not open. Enter it before you exit it.")
        try:
            self._own_states = [generator.get_state() for generator in self._generators]
            for generator, state in zip(self._generators, self._outer_states, strict=True):
                generator.set_state(state)
        finally:
            _current_seed_scope.reset(self._token)
            self._token = None


def fork_rng(scope: SeedScope) -> AbstractContextManager[None]:
    """Seeds the global generators from a seed scope while the context is open.

    The global generators are the torch CPU generator and the CUDA generator of the current device.
    On exit, the global generators return to the states they had before the context. The context can be entered
    again: each entry continues the numbers where the previous one stopped.

    Args:
        scope: The seed scope to take the seed from.

    Returns:
        A context manager that holds the fork while it is open.
    """
    return _Fork(scope)


def fork_rng_child(*keys: str | int) -> AbstractContextManager[None]:
    """Forks the global generators from a child of the seed scope of the innermost open ``fork_rng()``.

    Like ``fork_rng()``, the context can be entered again and then continues its numbers.

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
