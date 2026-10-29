from collections.abc import Iterable
from typing import cast

import torch
from torch.profiler import record_function

from d9d.core.dist_context import DistributedContext
from d9d.core.offload import DEFAULT_SLEEP_TAGS, Offloadable, OffloadContext, OnloadContext, SleepTag
from d9d.core.protocol import OptimizerProtocol
from d9d.loop.event import EventBus
from d9d.loop.event.catalogue.train import (
    EVENT_TRAIN_SLEEP_POST,
    EVENT_TRAIN_SLEEP_PRE,
    EVENT_TRAIN_WAKE_POST,
    EVENT_TRAIN_WAKE_PRE,
    EventSleepContext,
)

from .gradient_manager import GradientManager
from .model_stage_factory import TrackedModules


class TrainSleeper:
    """Component that offloads and restores the GPU-resident training state for a colocated RL hand-off.

    It runs the sleep/wake lifecycle for the ``Trainer``. It calls offload/onload on the ``Offloadable``
    subsystems in the correct order, surrounds the calls with lifecycle events and reports which
    subsystems are offloaded.
    """

    def __init__(
        self,
        dist_context: DistributedContext,
        tracked_modules: TrackedModules,
        optimizer: OptimizerProtocol,
        gradient_manager: GradientManager,
        event_bus: EventBus,
    ):
        """Constructs the ``TrainSleeper`` object.

        Args:
            dist_context: The distributed context.
            tracked_modules: Container of model parameters and buffers to offload.
            optimizer: The optimizer whose state is offloaded.
            gradient_manager: The gradient manager whose GPU buffers are released.
            event_bus: The event bus used to emit the sleep/wake lifecycle events.
        """
        self._dist_context = dist_context
        self._tracked_modules = tracked_modules
        self._optimizer = optimizer
        self._gradient_manager = gradient_manager
        self._event_bus = event_bus

    def sleep(self, tags: Iterable[SleepTag] = DEFAULT_SLEEP_TAGS) -> None:
        """Releases the GPU-resident training state selected by ``tags`` to host memory.

        This frees the GPU for a colocated workload, such as a rollout engine in colocated RL.
        The call is collective: every rank must invoke it with identical tags. Requesting a tag
        whose subsystem is already offloaded is a no-op.

        Args:
            tags: The subsystems to offload. Defaults to ``SleepTag.TENSOR_STATES``.

        Raises:
            NotImplementedError: If ``SleepTag.COMMS`` is requested. It is not implemented yet.
            RuntimeError: If called during an in-flight gradient accumulation.
        """
        with record_function("Sleep"):
            requested = frozenset(tags)
            if SleepTag.COMMS in requested:
                raise NotImplementedError(
                    "SleepTag.COMMS is not implemented yet. Only SleepTag.TENSOR_STATES is supported."
                )
            if SleepTag.TENSOR_STATES not in requested or self.is_sleeping(SleepTag.TENSOR_STATES):
                return

            if self._gradient_manager.has_in_flight_gradients:
                raise RuntimeError(
                    "Trainer.sleep() was called during an in-flight gradient accumulation. "
                    "Call it only between steps, e.g. from an EVENT_TRAIN_STEP_POST handler."
                )

            event_context = EventSleepContext(tags=requested)
            self._event_bus.trigger(EVENT_TRAIN_SLEEP_PRE, event_context)
            self._dist_context.wait_world()

            offload_context = OffloadContext(dist_context=self._dist_context, pin_memory=False)
            self._gradient_manager.offload(offload_context)
            cast(Offloadable, self._optimizer).offload(offload_context)
            self._tracked_modules.offload(offload_context)

            torch.cuda.synchronize(self._dist_context.current_device)
            torch.cuda.empty_cache()

            self._dist_context.wait_world()
            self._event_bus.trigger(EVENT_TRAIN_SLEEP_POST, event_context)

    def wake(self, tags: Iterable[SleepTag] = DEFAULT_SLEEP_TAGS) -> None:
        """Restores the training state released by ``sleep`` to the GPU.

        The call is collective: every rank must invoke it with identical tags. Requesting a tag
        whose subsystem is not offloaded is a no-op.

        Args:
            tags: The subsystems to restore. Defaults to ``SleepTag.TENSOR_STATES``.

        Raises:
            NotImplementedError: If ``SleepTag.COMMS`` is requested. It is not implemented yet.
        """
        with record_function("Wake"):
            requested = frozenset(tags)
            if SleepTag.COMMS in requested:
                raise NotImplementedError(
                    "SleepTag.COMMS is not implemented yet. Only SleepTag.TENSOR_STATES is supported."
                )
            if SleepTag.TENSOR_STATES not in requested or not self.is_sleeping(SleepTag.TENSOR_STATES):
                return

            event_context = EventSleepContext(tags=requested)
            self._event_bus.trigger(EVENT_TRAIN_WAKE_PRE, event_context)
            self._dist_context.wait_world()

            onload_context = OnloadContext(dist_context=self._dist_context)
            self._tracked_modules.onload(onload_context)
            cast(Offloadable, self._optimizer).onload(onload_context)
            self._gradient_manager.onload(onload_context)

            self._dist_context.wait_world()
            self._event_bus.trigger(EVENT_TRAIN_WAKE_POST, event_context)

    def is_sleeping(self, tag: SleepTag) -> bool:
        """Reports whether the subsystem identified by ``tag`` is offloaded.

        Args:
            tag: The subsystem to query.

        Returns:
            ``True`` if the subsystem is offloaded to host memory, otherwise ``False``.
        """
        if tag is SleepTag.TENSOR_STATES:
            return self._tracked_modules.is_offloaded()
        return False
