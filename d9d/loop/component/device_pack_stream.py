import abc
import collections
import dataclasses
from collections.abc import Iterator
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful

from d9d.core import pytree
from d9d.core.protocol import MicrobatchPackStream
from d9d.core.types import MicrobatchPack


def _copy_pack_to_device(pack: MicrobatchPack, device: torch.types.Device) -> MicrobatchPack:
    return [
        pytree.tree_map_only(torch.Tensor, lambda x: x.to(device, non_blocking=True), microbatch) for microbatch in pack
    ]


class DevicePackStream(abc.ABC, Stateful):
    """Hands the packs of a microbatch pack stream to the loop on the device; the data checkpoint boundary."""

    @abc.abstractmethod
    def __iter__(self) -> Iterator[MicrobatchPack]:
        """Yields the stream's packs on the device, ready for use on the current CUDA stream."""


class DirectDevicePackStream(DevicePackStream):
    """Copies each pack to the device on the current CUDA stream when it is handed out."""

    def __init__(self, stream: MicrobatchPackStream, device: torch.types.Device):
        """Constructs a DirectDevicePackStream object.

        Args:
            stream: The stream of host-side packs.
            device: The device to copy the packs to.
        """
        self._stream = stream
        self._device = device

    def __iter__(self) -> Iterator[MicrobatchPack]:
        for pack in self._stream:
            yield _copy_pack_to_device(pack, self._device)

    def state_dict(self) -> dict[str, Any]:
        return self._stream.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._stream.load_state_dict(state_dict)


@dataclasses.dataclass(frozen=True)
class _PrefetchedPack:
    pack: MicrobatchPack
    copied: torch.cuda.Event
    stream_state: dict[str, Any]


class PrefetchingDevicePackStream(DevicePackStream):
    """Copies packs to the device on a side CUDA stream ahead of the steps that consume them.

    Prefetching runs the stream's state ahead of the job, so ``state_dict`` returns the stream state as of the
    last handed out pack. This is the state of the last step, since checkpoints are only taken between steps.
    """

    def __init__(self, stream: MicrobatchPackStream, device: torch.types.Device, prefetch_factor: int):
        """Constructs a PrefetchingDevicePackStream object.

        Args:
            stream: The stream of host-side packs.
            device: The device to copy the packs to.
            prefetch_factor: The number of packs copied ahead.

        Raises:
            ValueError: If ``prefetch_factor`` is not positive.
        """
        if prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")

        self._stream = stream
        self._device = device
        self._prefetch_factor = prefetch_factor
        self._copy_stream = torch.cuda.Stream()

        self._prefetched: collections.deque[_PrefetchedPack] = collections.deque()
        self._handed_out_state: dict[str, Any] | None = None

    def _pull(self, packs: Iterator[MicrobatchPack]) -> _PrefetchedPack | None:
        pack = next(packs, None)
        if pack is None:
            return None

        stream_state = self._stream.state_dict()
        with torch.cuda.stream(self._copy_stream):
            device_pack = _copy_pack_to_device(pack, self._device)
            copied = torch.cuda.Event()
            copied.record()
        return _PrefetchedPack(pack=device_pack, copied=copied, stream_state=stream_state)

    def _hand_over(self, prefetched: _PrefetchedPack) -> MicrobatchPack:
        current_stream = torch.cuda.current_stream()
        current_stream.wait_event(prefetched.copied)
        # allocated on the copy stream, so keep the memory alive until the current stream is done with it
        for leaf in pytree.tree_leaves(prefetched.pack):
            if isinstance(leaf, torch.Tensor):
                leaf.record_stream(current_stream)

        self._handed_out_state = prefetched.stream_state
        return prefetched.pack

    def __iter__(self) -> Iterator[MicrobatchPack]:
        # exact until the first pack is pulled
        self._handed_out_state = self._stream.state_dict()
        try:
            packs = iter(self._stream)
            is_exhausted = False
            while True:
                while not is_exhausted and len(self._prefetched) <= self._prefetch_factor:
                    prefetched = self._pull(packs)
                    if prefetched is None:
                        is_exhausted = True
                    else:
                        self._prefetched.append(prefetched)

                if not self._prefetched:
                    return

                yield self._hand_over(self._prefetched.popleft())
        finally:
            self._prefetched.clear()

    def state_dict(self) -> dict[str, Any]:
        if self._handed_out_state is None:
            return self._stream.state_dict()
        return self._handed_out_state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._stream.load_state_dict(state_dict)
        self._handed_out_state = None


def build_device_pack_stream(
    stream: MicrobatchPackStream, device: torch.types.Device, prefetch_factor: int
) -> DevicePackStream:
    """Builds the device pack stream for the given prefetch factor.

    Args:
        stream: The stream of host-side packs.
        device: The device to copy the packs to.
        prefetch_factor: The number of packs copied ahead; ``0`` disables prefetching.

    Returns:
        A prefetching stream for a positive ``prefetch_factor``, a direct one for ``0``.

    Raises:
        ValueError: If ``prefetch_factor`` is negative.
    """
    if prefetch_factor < 0:
        raise ValueError("prefetch_factor must be non-negative")
    if prefetch_factor == 0:
        return DirectDevicePackStream(stream, device)
    return PrefetchingDevicePackStream(stream, device, prefetch_factor)
