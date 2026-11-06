from collections.abc import Iterator
from typing import Any

from d9d.data import MicrobatchPack, MicrobatchPackStream
from d9d.dist_context import BASE_DOMAIN, BaseDim, DistributedContext


class DataParallelMicrobatchPackStream(MicrobatchPackStream):
    """Microbatch pack stream wrapper that namespaces the checkpoint state per data-parallel rank."""

    def __init__(self, dist_context: DistributedContext, inner: MicrobatchPackStream):
        """Constructs the ``DataParallelMicrobatchPackStream`` object.

        Args:
            dist_context: The distributed context.
            inner: The wrapped microbatch pack stream that owns the actual data position.
        """
        self._dp_rank = dist_context.coordinate(BASE_DOMAIN, BaseDim.DP)
        self._inner = inner

    def __iter__(self) -> Iterator[MicrobatchPack]:
        return iter(self._inner)

    @property
    def total_steps(self) -> int | None:
        return self._inner.total_steps

    def state_dict(self) -> dict[str, Any]:
        return {f"dp_{self._dp_rank}": self._inner.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._inner.load_state_dict(state_dict[f"dp_{self._dp_rank}"])
