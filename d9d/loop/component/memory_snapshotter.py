from collections.abc import Generator
from contextlib import contextmanager

from d9d.core.dist_context import DistributedContext
from d9d.internals.profiling import MemorySnapshotter
from d9d.loop.config import MemorySnapshotConfig

from .job_schedule import JobSchedule


class ConfigurationMemorySnapshotter:
    """Manages the memory snapshot of the job configuration phase."""

    def __init__(self, dist_context: DistributedContext, config: MemorySnapshotConfig | None):
        """Constructs ConfigurationMemorySnapshotter object.

        Args:
            dist_context: The distributed context.
            config: Configuration settings for memory snapshots.
        """
        if config is None or not config.enabled or not config.configure:
            self._snapshotter = None
        else:
            self._snapshotter = MemorySnapshotter(
                save_dir=config.snapshots_dir, max_entries=config.max_entries, dist_context=dist_context
            )

    @contextmanager
    def record(self) -> Generator[None]:
        """Context manager recording the configuration phase into the ``configure`` snapshot.

        The snapshot is saved even if the configuration fails, so an out-of-memory failure
        during model construction leaves the snapshot of the allocations that caused it.
        """
        if self._snapshotter is None:
            yield
        else:
            with self._snapshotter.record("configure"):
                yield


class JobMemorySnapshotter:
    """Manages periodic memory snapshots of the job loop steps.

    The first ``active_steps`` steps of every ``period_steps``-long cycle of the global step are
    recorded into a single snapshot named after the number of steps completed by its end. A
    recording interrupted by the end of the loop or by an exception is saved as well.
    """

    def __init__(self, dist_context: DistributedContext, config: MemorySnapshotConfig | None, schedule: JobSchedule):
        """Constructs JobMemorySnapshotter object.

        Args:
            dist_context: The distributed context.
            config: Configuration settings for memory snapshots.
            schedule: Object tracking the current global step of the job loop.
        """
        if config is None or not config.enabled or config.steps is None:
            self._snapshotter = None
            self._steps = None
        else:
            self._snapshotter = MemorySnapshotter(
                save_dir=config.snapshots_dir, max_entries=config.max_entries, dist_context=dist_context
            )
            self._steps = config.steps
        self._schedule = schedule

        self._is_open = False
        self._is_recording = False

    @contextmanager
    def open(self) -> Generator[None]:
        """Context manager to activate memory snapshots for the job loop.

        Recording starts immediately if the current step belongs to a recorded window.

        Raises:
            RuntimeError: If the snapshotter is already open.
        """
        if self._is_open:
            raise RuntimeError("Memory snapshotter is already open")

        self._is_open = True
        try:
            self._start_if_recorded(self._schedule.current_step)
            yield
        finally:
            self._is_open = False
            self._dump_if_recording(self._schedule.current_step)

    def step(self):
        """Advances the snapshotting schedule at the end of a step.

        Must be called before the job schedule is advanced. Saves the snapshot if the current
        step closes a recorded window and starts recording if the next step opens one.

        Raises:
            RuntimeError: If called outside of the ``open`` context.
        """
        if not self._is_open:
            raise RuntimeError("Memory snapshotter must be open to step")

        if self._steps is None:
            return

        current_step = self._schedule.current_step

        if current_step % self._steps.period_steps == self._steps.active_steps - 1:
            self._dump_if_recording(current_step + 1)

        self._start_if_recorded(current_step + 1)

    def _start_if_recorded(self, step: int):
        if self._snapshotter is None or self._steps is None or self._is_recording:
            return

        if step >= self._schedule.total_steps or step % self._steps.period_steps >= self._steps.active_steps:
            return

        self._snapshotter.start()
        self._is_recording = True

    def _dump_if_recording(self, completed_steps: int):
        if self._snapshotter is None or not self._is_recording:
            return

        self._is_recording = False
        self._snapshotter.dump_and_stop(f"step_{completed_steps}")
