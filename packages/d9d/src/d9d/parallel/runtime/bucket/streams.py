import torch

from .buffer import PendingGradientBuffers


class SideStreams:
    """The streams that gather weights and reduce gradients, with the gradient buffers that the reduce stream reads."""

    def __init__(self):
        """Constructs the ``SideStreams`` object."""
        self._gather = torch.cuda.Stream()
        self._reduce = torch.cuda.Stream()
        self._pending = PendingGradientBuffers()

    @property
    def gather(self) -> torch.cuda.Stream:
        """The stream that gathers weights."""
        return self._gather

    @property
    def reduce(self) -> torch.cuda.Stream:
        """The stream that reduces gradients."""
        return self._reduce

    @property
    def pending(self) -> PendingGradientBuffers:
        """The gradient buffers that the reduce stream may still read."""
        return self._pending

    def wait_reductions(self):
        """Makes the current stream wait for the queued reductions, and frees the gradient buffers that they read."""
        torch.cuda.current_stream().wait_stream(self._reduce)
        self._pending.free_all()

    def close(self):
        """Makes the current stream wait for both side streams, and frees the pending gradient buffers."""
        torch.cuda.current_stream().wait_stream(self._gather)
        self.wait_reductions()
