from collections.abc import Generator
from contextlib import contextmanager

import torch.profiler

from d9d.core.dist_context import DistributedContext
from d9d.internals.profiling import Profiler
from d9d.loop.config import ProfilingConfig

from .job_schedule import JobSchedule


class JobProfiler:
    """Manages profiling sessions during a job loop.

    The profiling window follows the current step of the job schedule.
    """

    def __init__(self, dist_context: DistributedContext, config: ProfilingConfig | None, schedule: JobSchedule):
        """Constructs the ``JobProfiler`` object.

        Args:
            dist_context: The distributed context.
            config: The profiling configuration. ``None`` disables profiling.
            schedule: The schedule that tracks the current step of the loop.
        """
        self._config = config
        if config is None or not config.enabled:
            self._profiler = None
        else:
            self._profiler = Profiler(
                save_dir=config.traces_dir,
                active_steps=config.active_steps,
                warmup_steps=config.warmup_steps,
                period_steps=config.period_steps,
                record_shapes=config.record_shapes,
                with_stack=config.with_stack,
                dist_context=dist_context,
            )
        self._schedule = schedule

    @contextmanager
    def open(self) -> Generator[torch.profiler.profile | None]:
        """Activates profiling for the job loop.

        Yields:
            The active profiler if profiling is enabled, otherwise ``None``.
        """
        if self._profiler is None:
            yield None
        else:
            with self._profiler.open(self._schedule.current_step) as prof:
                yield prof
