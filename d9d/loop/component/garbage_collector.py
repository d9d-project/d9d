import gc
import time
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Self

from torch.profiler import record_function

from d9d.core.dist_context import DistributedContext
from d9d.loop.config import GarbageCollectionConfig

from .job_schedule import JobSchedule


class ManualGarbageCollector(AbstractContextManager):
    """Manages Python garbage collection during the training loop.

    This context manager disables automatic garbage collection on entry to avoid unpredictable
    latency spikes during steps. Collections then run only at configured intervals or when forced.
    """

    def __init__(self, dist_ctx: DistributedContext, config: GarbageCollectionConfig, schedule: JobSchedule):
        """Constructs the ``ManualGarbageCollector`` object.

        Args:
            dist_ctx: The distributed context.
            config: The configuration that sets how often garbage collection runs.
            schedule: The job schedule that tracks the current step.
        """
        self._dist_ctx = dist_ctx
        self._config = config
        self._schedule = schedule

    def __enter__(self) -> Self:
        """Disables automatic garbage collection and performs an initial full collection.

        Returns:
            The calling instance.
        """
        gc.disable()
        self._collect(generation=2)

        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None, /
    ) -> None:
        """Re-enables automatic garbage collection and performs a final full collection.

        Args:
            exc_type: The type of the exception raised (if any).
            exc_value: The exception instance raised (if any).
            traceback: The traceback object (if any).
        """
        gc.enable()
        self._collect(generation=2)

    def collect_periodic(self):
        """Collects generations 0 and 1 if the current step matches the configured period."""
        if self._schedule.should_do_action(self._config.period_steps, enable_on_last_step_if_periodic=False):
            self._collect(generation=1)

    def collect_forced(self):
        """Runs a full (generation 2) garbage collection regardless of the current step."""
        self._collect(generation=2)

    def _collect(self, generation: int):
        with record_function("Garbage Collection"):
            begin = time.monotonic()
            gc.collect(generation)
            end = time.monotonic()
            self._dist_ctx.logger.info(f"Garbage collection for generation {generation} took {end - begin:.2f} seconds")
