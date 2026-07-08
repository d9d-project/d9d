from collections.abc import Iterator
from contextlib import contextmanager
from typing import Generic, TypeVar, cast

import torch
import torch.utils._pytree as pytree  # noqa: PLC2701

from d9d.core.types import PyTree

TState = TypeVar("TState", bound=PyTree)

TMap = TypeVar("TMap")


def _detach_leaf(x: TMap) -> TMap:
    """Detaches a tensor from the computation graph if the input is a tensor.

    Args:
        x: The input object.

    Returns:
        The detached tensor or original object.
    """
    if isinstance(x, torch.Tensor):
        return cast(TMap, x.detach())
    return x


class PipelineStateHandler(Generic[TState]):
    """Holds the transient per-microbatch side-data of one step.

    Data returned by ``build_forward_inputs`` for microbatch ``i`` is stored here and read back when
    processing the outputs of the same microbatch ``i``. Stored state is detached from the autograd
    graph both when it is stored and when a ``scope`` closes, so caching it across the forward/backward
    never keeps the graph alive.
    """

    def __init__(self):
        """Constructs a PipelineStateHandler object."""
        self._state: dict[int, TState] = {}

    def store(self, microbatch_idx: int, state: TState):
        """Stores the (detached) side-data for a microbatch.

        Args:
            microbatch_idx: The index of the microbatch within the current pack.
            state: The side-data PyTree to store; every tensor leaf is detached.
        """
        self._state[microbatch_idx] = pytree.tree_map(_detach_leaf, state)

    @contextmanager
    def scope(self, microbatch_idx: int) -> Iterator[TState]:
        """Yields the stored side-data for a microbatch, detaching every tensor leaf on exit.

        Tensors written into the state while the scope is open are detached from the autograd
        graph when the block exits, so the cached state never keeps the graph alive after
        the microbatch is processed.

        Args:
            microbatch_idx: The index of the microbatch within the current pack.

        Yields:
            The side-data stored for this microbatch.
        """
        state = self._state[microbatch_idx]
        try:
            yield state
        finally:
            self._state[microbatch_idx] = pytree.tree_map(_detach_leaf, state)

    def reset(self):
        """Resets the underlying storage, clearing all state."""
        self._state.clear()
