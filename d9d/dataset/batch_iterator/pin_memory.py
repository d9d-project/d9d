from collections.abc import Iterator
from typing import Any

import torch

from d9d.core import pytree
from d9d.core.protocol import MicrobatchPackStream
from d9d.core.types import MicrobatchPack


def _pin_pack(pack: MicrobatchPack) -> MicrobatchPack:
    # a parallel host copy can spend most of its time synchronizing the intra-op thread pool, so pin with a single
    # thread, as the DataLoader's pin thread does; the setting is per thread and restored afterwards
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return [pytree.tree_map_only(torch.Tensor, lambda x: x.pin_memory(), microbatch) for microbatch in pack]
    finally:
        torch.set_num_threads(num_threads)


class PinMemoryMicrobatchPackStream(MicrobatchPackStream):
    """Wraps a microbatch pack stream to copy every tensor of each pack into page-locked (pinned) host memory.

    Pinned memory lets the loop copy packs to the device asynchronously. Unlike the ``pin_memory`` option of
    ``torch.utils.data.DataLoader``, the traversal uses the d9d pytree, so tensors nested in dataclasses are
    pinned as well. Pinning runs in the iterating thread.

    Its ``total_steps`` and state are those of the wrapped stream.
    """

    def __init__(self, inner: MicrobatchPackStream):
        """Constructs a PinMemoryMicrobatchPackStream object.

        Args:
            inner: The wrapped stream that owns the data and its position.
        """
        self._inner = inner

    def __iter__(self) -> Iterator[MicrobatchPack]:
        """Iterates the wrapped stream, pinning each pack.

        Yields:
            The next pack with all of its tensors in pinned host memory.
        """
        for pack in self._inner:
            yield _pin_pack(pack)

    @property
    def total_steps(self) -> int | None:
        """Returns the number of packs (steps) the wrapped stream yields.

        Returns:
            The step count of the wrapped stream, or ``None`` if it is unknown.
        """
        return self._inner.total_steps

    def state_dict(self) -> dict[str, Any]:
        return self._inner.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._inner.load_state_dict(state_dict)
