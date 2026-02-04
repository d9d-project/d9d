import gc
import time
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Self

from d9d.core.dist_context import DistributedContext
from d9d.loop.config import GarbageCollectionConfig

from .stepper import Stepper


class ManualGarbageCollector(AbstractContextManager):
    """
    Manages efficient Python garbage collection during the training loop.

    This context manager disables automatic garbage collection upon entry to prevent
    unpredictable latency spikes during training steps. It allows performing
    manual collection at specific intervals (periodic) or specific points (forced).
    """

    def __init__(
            self,
            dist_ctx: DistributedContext,
            config: GarbageCollectionConfig,
            step: Stepper
    ):
        """
        Constructs the garbage collector manager.

        Args:
            dist_ctx: The distributed context.
            config: Configuration determining how often GC should run.
            step: Stepper instance used to track the current training step.
        """
        self._dist_ctx = dist_ctx
        self._config = config
        self._step = step

    def __enter__(self) -> Self:
        """
        Disables automatic garbage collection and performs an initial full collection.

        Returns:
            The calling instance.
        """

        gc.disable()
        self._collect(generation=2)

        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None, /
    ) -> None:
        """
        Re-enables automatic garbage collection and performs a final full collection.

        Args:
            exc_type: The type of the exception raised (if any).
            exc_value: The exception instance raised (if any).
            traceback: The traceback object (if any).
        """

        gc.enable()
        self._collect(generation=2)

    def collect_periodic(self):
        """
        Triggers garbage collection if the current step matches the configured period.

        This typically performs a faster (generation 1) collection rather than a full sweep.
        """

        if self._step.should_do_action(self._config.period_steps, enable_on_last_step_if_periodic=False):
            self._collect(generation=1)

    def collect_forced(self):
        """
        Forces a full garbage collection run regardless of the step count.

        This performs a generation 2 collection.
        """

        self._collect(generation=2)

    def _collect(self, generation: int):
        begin = time.monotonic()
        gc.collect(generation)
        end = time.monotonic()
        self._dist_ctx.logger.info(f"[GC] Garbage collection for generation {generation} took {end - begin}s")
