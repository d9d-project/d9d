"""GPU memory that is counted until the GPU finishes reading it."""

from collections import deque

import torch
from torch import Tensor


class MemoryLedger:
    """Budget of managed memory, counted in GPU time.

    The allocator works in host time, but memory is free only when the GPU finishes reading it. So released
    memory stays busy here until its events fire, and the host waits only when the budget is exhausted.
    """

    def __init__(self, budget_bytes: int):
        """Constructs the ``MemoryLedger`` object.

        Args:
            budget_bytes: The budget of the managed memory.
        """
        self._budget = budget_bytes
        self._used = 0
        self._pending: deque[tuple[list[torch.cuda.Event], int]] = deque()

    def reserve(self, size: int):
        """Waits until the budget has room and takes it.

        Args:
            size: The size in bytes.

        Raises:
            ValueError: If live allocations leave insufficient room and no released allocation can be reclaimed.
        """
        if size > self._budget:
            raise ValueError(f"The allocation ({size}) does not fit into the memory budget ({self._budget}).")
        while self._used + size > self._budget:
            if not self._pending:
                raise ValueError(
                    f"The allocation ({size}) exceeds the available memory budget "
                    f"({self._budget - self._used}). Release a live buffer before allocating another."
                )
            events, released = self._pending.popleft()
            for event in events:
                event.synchronize()
            self._used -= released
        self._used += size

    def retire(self, events: list[torch.cuda.Event], size: int):
        """Marks memory as released once all events fire.

        Args:
            events: The events of the last reads.
            size: The size in bytes.
        """
        self._pending.append((events, size))


class ManagedBuffer:
    """A flat tensor whose ``Storage`` stays the same while its memory comes and goes.

    Saved autograd views point to the ``Storage``, so they stay valid. Replacing ``.data`` would break them.
    """

    def __init__(self, size: int, dtype: torch.dtype, home: torch.cuda.Stream, ledger: MemoryLedger):
        """Constructs the ``ManagedBuffer`` object. The memory is not held after construction.

        Args:
            size: The number of elements.
            dtype: The dtype.
            home: The stream that allocates the memory, so it lands in the allocator pool of that stream.
            ledger: The memory budget.
        """
        self._home = home
        self._ledger = ledger
        self._bytes = size * dtype.itemsize
        self._allocated = False
        with torch.cuda.stream(home):
            self.tensor: Tensor = torch.empty(size, dtype=dtype, device="cuda")
        self.tensor.untyped_storage().resize_(0)

    def acquire(self):
        """Gives the buffer uninitialized memory.

        Raises:
            ValueError: If the buffer already has memory.
        """
        if self._allocated:
            raise ValueError("The buffer already has memory. Call release() before acquire().")
        self._ledger.reserve(self._bytes)
        with torch.cuda.stream(self._home):
            self.tensor.untyped_storage().resize_(self._bytes)
        self._allocated = True

    @property
    def allocated(self) -> bool:
        """Whether the buffer currently owns memory."""
        return self._allocated

    def release(self, readers: list[torch.cuda.Stream]):
        """Takes the memory back after all queued reads.

        Args:
            readers: The streams that read the buffer. Without ``record_stream``, the allocator can give the
                block to another tensor while a reader of another stream still runs.

        Raises:
            ValueError: If the buffer has no memory.
        """
        if not self._allocated:
            raise ValueError("The buffer has no memory. Call acquire() before release().")
        events = []
        for stream in readers:
            self.tensor.record_stream(stream)
            events.append(stream.record_event())
        self.tensor.untyped_storage().resize_(0)
        self._ledger.retire(events, self._bytes)
        self._allocated = False
