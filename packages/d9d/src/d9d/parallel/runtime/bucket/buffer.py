from collections import deque

import torch
from torch import Tensor


class ManagedBuffer:
    """A flat tensor whose memory can be freed and given back while its views stay valid."""

    def __init__(self, size: int, dtype: torch.dtype, home: torch.cuda.Stream):
        """Constructs the ``ManagedBuffer`` object.

        The buffer has no memory until ``acquire()``.

        Args:
            size: The number of elements.
            dtype: The dtype of the elements.
            home: The stream that owns the memory.
        """
        self._home = home
        self._bytes = size * dtype.itemsize
        self._allocated = False
        with torch.cuda.stream(home):
            self._tensor = torch.empty(size, dtype=dtype, device="cuda")
        self._tensor.untyped_storage().resize_(0)

    @property
    def tensor(self) -> Tensor:
        """The flat tensor."""
        return self._tensor

    @property
    def allocated(self) -> bool:
        """Whether the buffer has memory."""
        return self._allocated

    def acquire(self):
        """Gives the buffer uninitialized memory.

        Raises:
            ValueError: If the buffer already has memory.
        """
        if self._allocated:
            raise ValueError("The buffer already has memory. Call release() before acquire().")
        # Resizing the Storage keeps all views of it. A new tensor or a .data swap would leave them on the old memory.
        with torch.cuda.stream(self._home):
            self._tensor.untyped_storage().resize_(self._bytes)
        self._allocated = True

    def require_writer(self):
        """Checks that the current stream is the home stream, the only one that may write the buffer.

        Raises:
            ValueError: If the current stream is another stream.
        """
        if torch.cuda.current_stream() != self._home:
            raise ValueError(
                f"The buffer belongs to the stream ({self._home}), but the current stream is "
                f"({torch.cuda.current_stream()}). Run bind() and the steps on one stream."
            )

    def wait(self, read: torch.cuda.Event):
        """Makes the home stream wait for a read on another stream.

        Args:
            read: The event after the read.
        """
        self._home.wait_event(read)

    def release(self, after: torch.cuda.Event | None):
        """Frees the memory, which the home stream can reuse at once.

        Args:
            after: The event after the last read on another stream, or ``None`` if no other stream read the buffer.

        Raises:
            ValueError: If the buffer has no memory.
        """
        if not self._allocated:
            raise ValueError("The buffer has no memory. Call acquire() before release().")
        if after is not None:
            self.wait(after)
        self._tensor.untyped_storage().resize_(0)
        self._allocated = False


# A wait for the reads right after the reduction would stall the backward pass behind the reductions. The reduce stream
# copies a gradient buffer behind the reduce-scatter of the previous bucket, so the copy ends about one bucket later.
# So the compute stream waits for it only two buckets later.
_PENDING_GRADIENT_BUFFERS = 2


class PendingGradientBuffers:
    """The gradient buffers that the reduce stream may still read, oldest first."""

    def __init__(self):
        """Constructs the ``PendingGradientBuffers`` object."""
        self._pending: deque[tuple[ManagedBuffer, torch.cuda.Event]] = deque()

    def push(self, buffer: ManagedBuffer, read: torch.cuda.Event):
        """Adds a buffer that the reduce stream reads.

        Args:
            buffer: The buffer.
            read: The event after the last read of the reduce stream.
        """
        self._pending.append((buffer, read))

    def take_back(self, buffer: ManagedBuffer) -> bool:
        """Removes a pending buffer with its memory, after its reads.

        Args:
            buffer: The buffer.

        Returns:
            Whether the buffer was pending.
        """
        for index, (pending, read) in enumerate(self._pending):
            if pending is buffer:
                del self._pending[index]
                buffer.wait(read)
                return True
        return False

    def make_room(self):
        """Frees the oldest buffers after their reads, until another buffer stays within the limit."""
        while len(self._pending) >= _PENDING_GRADIENT_BUFFERS:
            buffer, read = self._pending.popleft()
            buffer.release(after=read)

    def free_all(self):
        """Frees every pending buffer after its reads."""
        while self._pending:
            buffer, read = self._pending.popleft()
            buffer.release(after=read)
