import dataclasses

from d9d.loop.event import Event

from .common import (
    EventConfigurationStartedContext,
    EventDataLoaderReadyContext,
    EventModelStagesReadyContext,
    EventStepContext,
)


@dataclasses.dataclass(kw_only=True)
class EventInferenceReadyContext:
    """
    Context provided when inference is fully ready to begin and the checkpoint is loaded.
    """


@dataclasses.dataclass(kw_only=True)
class EventInferenceFinishedContext:
    """
    Context provided when the entire inference loop has completed successfully.
    """


# Configuration Events

EVENT_INFERENCE_CONFIG_STARTED = Event[EventConfigurationStartedContext](id="inference.configuration.start")
"""Triggered when the inference configuration process begins. Provides access to the distributed context."""

EVENT_INFERENCE_DATA_LOADER_READY = Event[EventDataLoaderReadyContext](id="inference.configuration.data_loader")
"""Triggered when the inference data loader has been fully initialized."""

EVENT_INFERENCE_MODEL_STAGES_READY = Event[EventModelStagesReadyContext](id="inference.configuration.model_stages")
"""Triggered when the model stages are initialized for inference."""


# Runtime Events

EVENT_INFERENCE_READY = Event[EventInferenceReadyContext](id="inference.ready")
"""Triggered right before the main inference loop starts, after configuration is complete and checkpoints are loaded."""

EVENT_INFERENCE_STEP_PRE = Event[EventStepContext](id="inference.step.pre")
"""Triggered at the absolute beginning of an inference step iteration."""

EVENT_INFERENCE_STEP_POST = Event[EventStepContext](id="inference.step.post")
"""Triggered at the very end of an inference step iteration (excluding checkpointing)."""

EVENT_INFERENCE_FORWARD_PRE = Event[EventStepContext](id="inference.forward.pre")
"""Triggered immediately before the forward pass sequence begins."""

EVENT_INFERENCE_FORWARD_POST = Event[EventStepContext](id="inference.forward.post")
"""Triggered immediately after the forward pass sequence for the current step has finished."""

EVENT_INFERENCE_FINISHED = Event[EventInferenceFinishedContext](id="inference.finished")
"""Triggered when the entire inference execution loop finishes successfully."""
