# Event Bus & Hooks

## About

`d9d` does not use a fixed set of lifecycle methods, such as `on_step_start` or `on_post_optimizer`. Instead, a typed **event bus** lets you extend both the training and the inference loops. Any user component (`TrainTask`, `ModelProvider`, etc.) can subscribe to specific points of the execution. You subscribe *only* to the events you need, so your code does not depend on the internal execution order.

## How It Works

The system has three core concepts:

1.  **`Event[TContext]`**: A typed descriptor of a specific moment in the lifecycle.
2.  **Contexts**: Data classes (e.g. `EventStepContext`) that hold the state relevant to the event.
3.  **`EventBus`**: The dispatcher that passes contexts to the subscribed handlers.

!!! note
    The event bus does not catch exceptions. If a handler raises, the loop stops immediately.

## Registering Handlers

`BaseTask` (and so `TrainTask` and `InferenceTask`) and `ModelProvider` have a `register_events` hook. The loop calls it during configuration and passes a context that holds the `EventBus`.

### Declarative Registration (Recommended)

Mark your methods with the `@subscribe` decorator. Then call `subscribe_annotated` to register all of them at once.

```python
from d9d.loop.control import RegisterTaskEventsContext, TrainTask
from d9d.loop.event import subscribe, subscribe_annotated
from d9d.loop.event.catalogue.common import EventModelStagesReadyContext, EventStepContext
from d9d.loop.event.catalogue.train import EVENT_TRAIN_MODEL_STAGES_READY, EVENT_TRAIN_STEP_POST


class CustomTrainTask(TrainTask):
    def __init__(self):
        self._modules = []

    def register_events(self, ctx: RegisterTaskEventsContext) -> None:
        # Registers every method of this instance marked with @subscribe.
        subscribe_annotated(ctx.event_bus, self)

    @subscribe(EVENT_TRAIN_MODEL_STAGES_READY)
    def _on_model_ready(self, ctx: EventModelStagesReadyContext) -> None:
        self._modules = ctx.modules

    @subscribe(EVENT_TRAIN_STEP_POST)
    def _on_step_post(self, ctx: EventStepContext) -> None:
        print(f"Step {ctx.schedule.current_step} of {ctx.schedule.total_steps} completed.")

    def compute_loss(self, ctx):
        ...  # Task logic
```

### Manual Registration

You can also call the `EventBus` directly. This is useful for simple lambda callbacks or handlers created at runtime.

```python
from d9d.loop.control import RegisterTaskEventsContext, TrainTask
from d9d.loop.event.catalogue.train import EVENT_TRAIN_OPTIMIZER_READY


class CustomTrainTask(TrainTask):
    def register_events(self, ctx: RegisterTaskEventsContext) -> None:
        ctx.event_bus.subscribe(
            EVENT_TRAIN_OPTIMIZER_READY,
            lambda event_ctx: print(f"Optimizer loaded: {event_ctx.optimizer}"),
        )
```

## Custom Events

You can define and trigger your own events in custom logic.

```python
import dataclasses
from pathlib import Path

from d9d.loop.event import Event, EventBus


@dataclasses.dataclass(kw_only=True)
class CheckpointContext:
    step: int
    path: Path


# Define a new event.
EVENT_CHECKPOINT_SAVED = Event[CheckpointContext](id="user.checkpoint_saved")


# Trigger it from your task.
def process_something(bus: EventBus):
    bus.trigger(
        EVENT_CHECKPOINT_SAVED,
        CheckpointContext(step=1000, path=Path("/checkpoints/step_1000")),
    )
```

## API Reference

### Core Components

::: d9d.loop.event
    options:
      heading_level: 4

### Common Events

::: d9d.loop.event.catalogue.common
    options:
      heading_level: 4

### Training Events

::: d9d.loop.event.catalogue.train
    options:
      heading_level: 4

### Inference Events

::: d9d.loop.event.catalogue.inference
    options:
      heading_level: 4
