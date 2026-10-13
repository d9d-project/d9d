import dataclasses
from collections import deque
from contextlib import AbstractContextManager
from types import TracebackType

import torch

from d9d.core import pytree
from d9d.tracker import BaseTrackerRun, TrackerPoint, TrackerValue

from .job_schedule import JobSchedule


@dataclasses.dataclass(kw_only=True)
class _PendingWrite:
    point: TrackerPoint
    value: TrackerValue
    copies: list[torch.cuda.Event]

    def is_copied(self) -> bool:
        return all(copy.query() for copy in self.copies)

    def wait_copied(self):
        for copy in self.copies:
            copy.synchronize()


def _copy_tensor_to_host(tensor: torch.Tensor) -> torch.Tensor:
    # A copy of a tensor that requires grad would keep its autograd graph alive until the write.
    tensor = tensor.detach()
    match tensor.device.type:
        case "cpu":
            # The caller can change the tensor in place after the call, so the write keeps its own copy.
            return tensor.clone()
        case "cuda":
            # A non-blocking copy to the host goes to pinned memory and does not wait for the GPU.
            return tensor.to("cpu", non_blocking=True)
        case _:
            raise ValueError(f"Device type ({tensor.device.type}) is not supported. Log CPU or CUDA tensors.")


def _copy_to_host(value: TrackerValue) -> tuple[TrackerValue, list[torch.cuda.Event]]:
    host_value = pytree.tree_map_only(torch.Tensor, _copy_tensor_to_host, value)

    # An event comes after every copy queued before it on its stream, so one event per device covers all copies.
    devices = {leaf.device for leaf in pytree.tree_leaves(value) if isinstance(leaf, torch.Tensor) and leaf.is_cuda}
    copies = []
    for device in devices:
        copy = torch.cuda.Event()
        copy.record(torch.cuda.current_stream(device))
        copies.append(copy)
    return host_value, copies


class PendingTrackerWrites(AbstractContextManager):
    """Queue of tracker writes that wait for their values to be copied from the GPU.

    A write reaches the tracker run once its copy has finished, in the order the writes were queued. A new write
    waits for the queued writes that are ``max_pending_steps`` or more steps behind it.

    The queue accepts writes while its context is open. On exit, it passes all queued writes to the run, or only the
    finished ones if the context exits with an exception.
    """

    def __init__(self, run: BaseTrackerRun, max_pending_steps: int):
        """Constructs the ``PendingTrackerWrites`` object.

        Args:
            run: The tracker run to write to.
            max_pending_steps: How many steps the queued writes can lag behind the newest write.
        """
        self._run = run
        self._max_pending_steps = max_pending_steps
        self._writes: deque[_PendingWrite] = deque()
        self._closed = False

    def put(self, write: _PendingWrite):
        """Queues a write and passes the finished writes to the tracker run.

        Args:
            write: The write to queue.

        Raises:
            RuntimeError: If the queue is closed.
        """
        if self._closed:
            raise RuntimeError("The tracker run is closed. Log values only while the run is open.")

        self._writes.append(write)
        self._drain()

        # Waiting for an earlier step also limits how far the host runs ahead of the GPU.
        while self._writes and write.point.step - self._writes[0].point.step >= self._max_pending_steps:
            self._write_oldest()

    def _drain(self):
        while self._writes and self._writes[0].is_copied():
            self._write_oldest()

    def flush(self):
        """Waits for all queued copies and passes every queued write to the tracker run."""
        while self._writes:
            self._write_oldest()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ):
        self._closed = True
        if exc_type is None:
            self.flush()
            return

        # Waiting for the GPU after an exception could hang on a stuck GPU or raise again after a CUDA error.
        try:
            for write in self._writes:
                if write.is_copied():
                    self._run.write(write.point, write.value)
        finally:
            self._writes.clear()

    def _write_oldest(self):
        write = self._writes.popleft()
        write.wait_copied()
        self._run.write(write.point, write.value)


class TrackerWriter:
    """Writer that logs values, including GPU tensors, to a tracker run without waiting for the GPU.

    A call reads a GPU tensor on the current CUDA stream, so later in-place changes on that stream do not affect
    the logged value. The value reaches the tracker after the GPU has computed it, at a later call of the writer or
    when the run closes. Each value gets the step that is current at the call.
    """

    def __init__(self, pending: PendingTrackerWrites, schedule: JobSchedule, context: dict[str, str]):
        """Constructs the ``TrackerWriter`` object.

        Args:
            pending: The queue that passes the writes to the tracker run.
            schedule: The job schedule whose current step is attached to a value at the call.
            context: The tags attached to every value, e.g. ``{"stage": "train"}``.
        """
        self._pending = pending
        self._schedule = schedule
        self._context = context

    def log(self, name: str, value: TrackerValue, context: dict[str, str] | None = None):
        """Logs a value.

        Args:
            name: The name of the series.
            value: The value. Its tensors can be on any device.
            context: Tags for this value only. They override the tags of the writer with the same name.
        """
        host_value, copies = _copy_to_host(value)
        self._pending.put(_PendingWrite(point=self._point(name, context), value=host_value, copies=copies))

    def _point(self, name: str, context: dict[str, str] | None) -> TrackerPoint:
        # Each point owns its context, so a change by a tracker run does not leak into other points.
        return TrackerPoint(name=name, step=self._schedule.current_step, context={**self._context, **(context or {})})
