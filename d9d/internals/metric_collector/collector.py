import torch
from torch.profiler import record_function

from d9d.core import pytree
from d9d.core.dist_context import DistributedContext
from d9d.core.types import PyTree
from d9d.metric import Metric


class AsyncMetricCollector:
    """Synchronizes and computes a metric asynchronously on a side CUDA stream.

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

    def schedule_collection(self, dist_context: DistributedContext):
        """Schedules the metric synchronization and computation on the side stream.

        In a distributed setup, the metric is synchronized across ranks before the computation.

        Args:
            dist_context: The distributed context to synchronize the metric with.

        Raises:
            RuntimeError: If the collector is not bound.
        """
        if self._stream is None:
            raise RuntimeError("AsyncMetricCollector is not bound. Call bind() first.")

        # The side stream must see the metric updates queued on the current stream.
        self._stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self._stream), record_function("Async Metric Sync & Compute"):
            if dist_context.mesh_params.is_distributed:
                self._metric.sync(dist_context)
            self._compute_buffer = self._metric.compute()

    def collect_results(self) -> PyTree[float | int | bool]:
        """Waits for the asynchronous computation, returns its results and resets the metric.

        Returns:
            A PyTree with the structure of the metric output and Python scalars (``float``, ``int`` or ``bool``)
            as leaves.

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

        # Sync to CPU.
        results = pytree.tree_map(lambda x: x.cpu(), results)
        results = pytree.tree_map(lambda x: x.item(), results)

        # Safe: the current stream already waited for the side stream that read the metric state.
        self._metric.reset()

        return results
