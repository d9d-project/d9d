from collections.abc import Iterable
from typing import cast

import torch

from d9d.core.offload import DEFAULT_SLEEP_TAGS, Offloadable, OffloadContext, OnloadContext, SleepTag
from d9d.loop.event.catalogue.train import (
    EVENT_TRAIN_SLEEP_POST,
    EVENT_TRAIN_SLEEP_PRE,
    EVENT_TRAIN_WAKE_POST,
    EVENT_TRAIN_WAKE_PRE,
    EventSleepContext,
)
from d9d.loop.state import TrainJobState


class TrainSleeper:
    """
    Offloads and restores the GPU-resident training state for a colocated RL hand-off.

    This encapsulates the sleep/wake lifecycle on behalf of the "Trainer": it fans the
    offload/onload calls out to the "Offloadable" subsystems in the correct order, surrounds
    them with the lifecycle events, and reports which subsystems are currently offloaded.
    """

    def __init__(self, state: TrainJobState):
        """
        Constructs a TrainSleeper bound to a training job state.

        Args:
            state: The training job state whose subsystems are offloaded and restored.
        """
        self._state = state

    def sleep(self, tags: Iterable[SleepTag] = DEFAULT_SLEEP_TAGS) -> None:
        """
        Releases the GPU-resident training state selected by "tags" to host memory.

        This frees the GPU for a colocated workload, such as a rollout engine in colocated RL.
        The call is collective: every rank must invoke it with identical tags. Requesting a tag
        whose subsystem is already offloaded is a no-op.

        Args:
            tags: The subsystems to offload. Defaults to "SleepTag.TENSOR_STATES".

        Raises:
            NotImplementedError: If "SleepTag.COMMS" is requested, since it is not yet implemented.
            RuntimeError: If called during an in-flight gradient accumulation.
        """

        requested = frozenset(tags)
        if SleepTag.COMMS in requested:
            raise NotImplementedError(
                "SleepTag.COMMS is not yet implemented. Only SleepTag.TENSOR_STATES is supported."
            )
        if SleepTag.TENSOR_STATES not in requested or self.is_sleeping(SleepTag.TENSOR_STATES):
            return

        if self._state.gradient_manager.has_in_flight_gradients:
            raise RuntimeError(
                "Trainer.sleep() was called during an in-flight gradient accumulation. "
                "Sleep is only legal between steps, e.g. from an EVENT_TRAIN_STEP_POST handler."
            )

        event_context = EventSleepContext(tags=requested)
        self._state.event_bus.trigger(EVENT_TRAIN_SLEEP_PRE, event_context)
        self._state.dist_context.wait_world()

        offload_context = OffloadContext(dist_context=self._state.dist_context, pin_memory=False)
        self._state.gradient_manager.offload(offload_context)
        cast(Offloadable, self._state.optimizer).offload(offload_context)
        self._state.tracked_modules.offload(offload_context)

        torch.cuda.synchronize(self._state.dist_context.current_device)
        torch.cuda.empty_cache()

        self._state.dist_context.wait_world()
        self._state.event_bus.trigger(EVENT_TRAIN_SLEEP_POST, event_context)

    def wake(self, tags: Iterable[SleepTag] = DEFAULT_SLEEP_TAGS) -> None:
        """
        Restores GPU residency of the training state previously released by "sleep".

        The call is collective: every rank must invoke it with identical tags. Requesting a tag
        whose subsystem is not offloaded is a no-op.

        Args:
            tags: The subsystems to restore. Defaults to "SleepTag.TENSOR_STATES".

        Raises:
            NotImplementedError: If "SleepTag.COMMS" is requested, since it is not yet implemented.
        """

        requested = frozenset(tags)
        if SleepTag.COMMS in requested:
            raise NotImplementedError(
                "SleepTag.COMMS is not yet implemented. Only SleepTag.TENSOR_STATES is supported."
            )
        if SleepTag.TENSOR_STATES not in requested or not self.is_sleeping(SleepTag.TENSOR_STATES):
            return

        event_context = EventSleepContext(tags=requested)
        self._state.event_bus.trigger(EVENT_TRAIN_WAKE_PRE, event_context)
        self._state.dist_context.wait_world()

        onload_context = OnloadContext(dist_context=self._state.dist_context)
        self._state.tracked_modules.onload(onload_context)
        cast(Offloadable, self._state.optimizer).onload(onload_context)
        self._state.gradient_manager.onload(onload_context)

        self._state.dist_context.wait_world()
        self._state.event_bus.trigger(EVENT_TRAIN_WAKE_POST, event_context)

    def is_sleeping(self, tag: SleepTag) -> bool:
        """
        Reports whether the subsystem identified by "tag" is currently offloaded.

        Args:
            tag: The subsystem to query.

        Returns:
            True if the subsystem is offloaded to host memory, False otherwise.
        """

        if tag is SleepTag.TENSOR_STATES:
            return self._state.tracked_modules.is_offloaded()
        return False
