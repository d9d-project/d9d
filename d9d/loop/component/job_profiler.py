from collections.abc import Generator
from contextlib import contextmanager

import torch.profiler

from d9d.core.dist_context import DistributedContext
from d9d.internals.profiling import Profiler
from d9d.loop.config import ProfilingConfig

from .stepper import Stepper


class JobProfiler:
    """
    Manages profiling sessions during a job loop.

    This class coordinates the initialization and activation of the internal
    profiler based on the current step count provided by the stepper.
    """

    def __init__(self, dist_context: DistributedContext, config: ProfilingConfig | None, stepper: Stepper):
        """
        Constructs JobProfiler object.

        Args:
            dist_context: The distributed context.
            config: Configuration settings for profiling.
            stepper: Object tracking the current global step of the training loop.
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
                dist_context=dist_context,
            )
        self._stepper = stepper

    @contextmanager
    def open(self) -> Generator[torch.profiler.profile | None]:
        """
        Context manager to activate profiling for the job loop.

        Yields:
            The active Profiler instance if profiling is enabled, otherwise None.
        """

        if self._profiler is None:
            yield None
        else:
            with self._profiler.open(self._stepper.current_step) as prof:
                yield prof
