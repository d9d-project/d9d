import torch
from torch.distributed import ProcessGroup
from torch.profiler import record_function

from d9d import pytree
from d9d.pytree import PyTree

from .abc import Metric


class AsyncMetricCollector:
    """Asynchronous metric collector that synchronizes and computes a metric on a side CUDA stream.

    The distributed reduction and the computation run on the side stream, off the main training stream.
    """

    def __init__(self, metric: Metric):
        """Constructs the ``AsyncMetricCollector`` object.

        Args:
            metric: The metric to collect.
        """
        self._metric = metric
        self._stream: torch.cuda.Stream | None = None
        self._compute_buffer: PyTree[torch.Tensor] | None = None

    def bind(self):
        """Moves the metric to CUDA and creates the side stream."""
        self._metric.to("cuda")
        self._stream = torch.cuda.Stream()

    def unbind(self):
        """Releases the reference to the side stream."""
        self._stream = None

    def schedule_collection(self, group: ProcessGroup | None):
        """Schedules the metric synchronization and computation on the side stream.

        Args:
            group: The process group to synchronize the metric over before the computation. ``None`` skips the
                synchronization, e.g. in a single process.

        Raises:
            RuntimeError: If the collector is not bound.
        """
        if self._stream is None:
            raise RuntimeError("AsyncMetricCollector is not bound. Call bind() first.")

        # The side stream must see the metric updates queued on the current stream.
        self._stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self._stream), record_function("Async Metric Sync & Compute"):
            if group is not None:
                self._metric.sync(group)
            # compute() can return the metric state itself, which collect_results() resets.
            self._compute_buffer = pytree.tree_map(lambda x: x.clone(), self._metric.compute())

    def collect_results(self) -> PyTree[torch.Tensor]:
        """Returns the results of the asynchronous computation and resets the metric.

        The current CUDA stream waits for the computation, the host does not.

        Returns:
            A PyTree with the structure of the metric output and tensors on the device of the metric as leaves.

        Raises:
            RuntimeError: If the collector is not bound, or if ``schedule_collection`` was not called before.
        """
        if self._stream is None:
            raise RuntimeError("AsyncMetricCollector is not bound. Call bind() first.")

        if self._compute_buffer is None:
            raise RuntimeError("schedule_collection() was not called. Call it before collect_results().")

        torch.cuda.current_stream().wait_stream(self._stream)
        results = self._compute_buffer
        self._compute_buffer = None

        # Safe: the current stream already waited for the side stream that read the metric state.
        self._metric.reset()

        return results
