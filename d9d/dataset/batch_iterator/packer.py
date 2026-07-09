import math
from collections.abc import Iterator
from typing import Any

from d9d.core.protocol import DataLoaderProtocol, MicrobatchPackStream
from d9d.core.types import MicrobatchPack, PyTree


class FixedCountMicrobatchPacker(MicrobatchPackStream):
    """The default ``MicrobatchPackStream`` that groups ``microbatches_per_step`` microbatches from a loader per pack.

    Keeping a short trailing pack (``drop_last=False``) is only consistent across ranks when every rank sees the
    same number of microbatches (e.g. the dataset was sharded with ``pad_to_equal_size_across_shards``).

    Its ``total_steps`` is derived from the loader's length, and it delegates its state to the loader
    (the checkpoint boundary).
    """

    def __init__(self, loader: DataLoaderProtocol, microbatches_per_step: int, drop_last: bool = True):
        """Constructs a FixedCountMicrobatchPacker object.

        Args:
            loader: The microbatch stream to group.
            microbatches_per_step: The number of microbatches in each pack.
            drop_last: Whether to drop the trailing incomplete pack instead of yielding it short.

        Raises:
            ValueError: If ``microbatches_per_step`` is not positive.
        """
        if microbatches_per_step <= 0:
            raise ValueError("microbatches_per_step must be positive")

        self._loader = loader
        self._microbatches_per_step = microbatches_per_step
        self._drop_last = drop_last

    def __iter__(self) -> Iterator[MicrobatchPack]:
        """Iterates the loader, grouping microbatches into packs.

        Yields:
            A full pack, or a shorter trailing pack when ``drop_last`` is unset.
        """
        pack: list[PyTree] = []
        for microbatch in self._loader:
            pack.append(microbatch)
            if len(pack) == self._microbatches_per_step:
                yield pack
                pack = []

        if pack and not self._drop_last:
            yield pack

    @property
    def total_steps(self) -> int | None:
        """Returns the number of packs (steps) this packer yields.

        Returns:
            The pack count, including a trailing short pack unless ``drop_last`` is set.
        """
        if self._drop_last:
            return len(self._loader) // self._microbatches_per_step
        return math.ceil(len(self._loader) / self._microbatches_per_step)

    def state_dict(self) -> dict[str, Any]:
        return self._loader.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._loader.load_state_dict(state_dict)
