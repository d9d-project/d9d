import dataclasses

from d9d.core.protocol import LRSchedulerProtocol, OptimizerProtocol
from d9d.loop.event import Event
from d9d.tracker import BaseTrackerRun

from .common import (
    EventConfigurationStartedContext,
    EventDataLoaderReadyContext,
    EventModelStagesReadyContext,
    EventStepContext,
)


@dataclasses.dataclass(kw_only=True)
class EventOptimizerReadyContext:
    """
    Context provided when the optimizer has been instantiated.

    Attributes:
        optimizer: The optimizer instance wrapping the model parameters.
    """

    optimizer: OptimizerProtocol


@dataclasses.dataclass(kw_only=True)
class EventLRSchedulerReadyContext:
    """
    Context provided when the learning rate scheduler has been instantiated.

    Attributes:
        lr_scheduler: The learning rate scheduler instance.
    """

    lr_scheduler: LRSchedulerProtocol


@dataclasses.dataclass(kw_only=True)
class EventTrainReadyContext:
    """
    Context provided when training is fully ready to begin and the checkpoint is loaded.
    """

    run: BaseTrackerRun


@dataclasses.dataclass(kw_only=True)
class EventTrainFinishedContext:
    """
    Context provided when the entire training loop has completed successfully.
    """


# Configuration Events

EVENT_TRAIN_CONFIG_STARTED = Event[EventConfigurationStartedContext](id="train.configuration.start")
"""Triggered when the training configuration process begins. Provides access to the distributed context."""

EVENT_TRAIN_DATA_LOADER_READY = Event[EventDataLoaderReadyContext](id="train.configuration.data_loader")
"""Triggered when the training data loader has been fully initialized."""

EVENT_TRAIN_MODEL_STAGES_READY = Event[EventModelStagesReadyContext](id="train.configuration.model_stages")
"""Triggered when the model stages are initialized and parallelized."""

EVENT_TRAIN_OPTIMIZER_READY = Event[EventOptimizerReadyContext](id="train.configuration.optimizer")
"""Triggered when the optimizer has been built and is ready for use."""

EVENT_TRAIN_LR_SCHEDULER_READY = Event[EventLRSchedulerReadyContext](id="train.configuration.lr_scheduler")
"""Triggered when the learning rate scheduler has been configured."""


# Runtime Events

EVENT_TRAIN_READY = Event[EventTrainReadyContext](id="train.ready")
"""Triggered right before the main loop starts, after configuration is complete and checkpoints are loaded."""

EVENT_TRAIN_STEP_PRE = Event[EventStepContext](id="train.step.pre")
"""Triggered at the absolute beginning of a training step iteration."""

EVENT_TRAIN_STEP_POST = Event[EventStepContext](id="train.step.post")
"""Triggered at the very end of a training step iteration, after all operations (excluding checkpointing)."""

EVENT_TRAIN_FORWARD_BACKWARD_PRE = Event[EventStepContext](id="train.forward_backward.pre")
"""Triggered immediately before the sequence of forward and backward passes begins."""

EVENT_TRAIN_FORWARD_BACKWARD_POST = Event[EventStepContext](id="train.forward_backward.post")
"""Triggered immediately after all forward and backward passes for the current step have finished."""

EVENT_TRAIN_OPTIMIZER_STEP_PRE = Event[EventStepContext](id="train.optimizer_step.pre")
"""Triggered immediately before the optimizer updates the model parameters (but after gradients are scaled/clipped)."""

EVENT_TRAIN_OPTIMIZER_STEP_POST = Event[EventStepContext](id="train.optimizer_step.post")
"""Triggered immediately after the optimizer has updated the model parameters but before gradients are zeroed."""

EVENT_TRAIN_FINISHED = Event[EventTrainFinishedContext](id="train.finished")
"""Triggered when the entire training execution loop finishes successfully."""
