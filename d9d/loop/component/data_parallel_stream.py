from collections.abc import Iterator
from typing import Any

from d9d.core.dist_context import BATCH_DOMAIN, DistributedContext
from d9d.core.protocol import MicrobatchPackStream
from d9d.core.types import MicrobatchPack


class DataParallelMicrobatchPackStream(MicrobatchPackStream):
    """Wraps a microbatch pack stream to namespace its checkpoint state per data-parallel rank."""

    def __init__(self, dist_context: DistributedContext, inner: MicrobatchPackStream):
        """Constructs a DataParallelMicrobatchPackStream.

        Args:
            dist_context: The distributed context.
            inner: The wrapped microbatch pack stream that owns the actual data position.
        """
        if dist_context.mesh_params.is_distributed:
            self._dp_rank = dist_context.mesh_for(BATCH_DOMAIN)["dp"].get_local_rank()
        else:
            self._dp_rank = 0
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
