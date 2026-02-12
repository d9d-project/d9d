import torch
import torch.utils._pytree as pytree  # noqa: PLC2701
from torch.profiler import record_function

from d9d.core.dist_context import DistributedContext
from d9d.core.types import PyTree
from d9d.metric import Metric


class AsyncMetricCollector:
    """Helper class to synchronize and compute metrics asynchronously on a separate CUDA stream.

    This class decouples metric synchronization and computation from the main training loop.
    It schedules the heavy lifting (distributed reduction and tensor operations) on a
    secondary stream.
    """

    def __init__(self, metric: Metric):
        """Constructs AsyncMetricCollector object.

        Args:
            metric: The metric instance to collect and compute asynchronously.
        """
        self._metric = metric
        self._stream: torch.cuda.Stream | None = None
        self._compute_buffer: PyTree[torch.Tensor] | None = None

    def bind(self):
        """Moves the underlying metric to CUDA and initializes the side stream."""
        self._metric.to("cuda")
        self._stream = torch.cuda.Stream()

    def unbind(self):
        """Releases the reference to the side stream."""
        self._stream = None

    def schedule_collection(self, dist_context: DistributedContext):
        """Schedules metric synchronization and computation on the side stream.

        This method records a dependency on the current stream to ensure all data
        required for the metric is available, then launches the synchronization
        (if distributed) and computation tasks on the dedicated side stream.

        Args:
            dist_context: Distributed context used for metric synchronization across ranks.

        Raises:
            RuntimeError: If the collector has not been bound via .bind().
        """

        if self._stream is None:
            raise RuntimeError("AsyncMetricSynchronizer is not bound. Call .bind() first.")

        # depend on main stream
        self._stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self._stream), record_function("Async Metric Sync & Compute"):
            if dist_context.mesh_params.is_distributed:
                self._metric.sync(dist_context)
            self._compute_buffer = self._metric.compute()

    def collect_results(self) -> PyTree[float | int | bool]:
        """Waits for the async computation to finish and retrieves results.

        This method synchronizes the current stream with the side stream, moves
        results to CPU, converts them to Python scalars, and resets the underlying metric.

        Returns:
            A PyTree structure matching the metric's output containing python scalars
            (float, int, or bool) located on the CPU.

        Raises:
            RuntimeError: If the collector is not bound or if schedule_collection
                was not called prior to this method.
        """

        if self._stream is None:
            raise RuntimeError("AsyncMetricSynchronizer is not bound. Call .bind() first.")

        if self._compute_buffer is None:
            raise RuntimeError("sync_and_compute() was not called.")

        # wait for synchronization and computation to finish
        torch.cuda.current_stream().wait_stream(self._stream)
        results = self._compute_buffer
        self._compute_buffer = None

        # sync to CPU
        results = pytree.tree_map(lambda x: x.cpu(), results)
        results = pytree.tree_map(lambda x: x.item(), results)

        # reset on GPU safely
        self._metric.reset()

        return results
