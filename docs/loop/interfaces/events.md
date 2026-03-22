# Event Bus & Hooks

## Overview

Instead of hardcoding a fixed set of lifecycle methods (like `on_step_start` or `on_post_optimizer`), `d9d` uses a typed **Event Bus** for extending both the training and inference loops.

This publish-subscribe mechanism allows any user component (`TrainTask`, `ModelProvider`, etc.) to hook into specific execution points natively. You subscribe *only* to the events you care about, keeping your code clean and decoupled from the framework's internal execution order.

## How it Works

The system revolves around three core concepts:

1. **`Event[TContext]`**: A lightweight, typed descriptor representing a specific moment in the lifecycle.
2. **Contexts**: Data Classes (e.g., `EventStepContext`) holding the state relevant to the event.
3. **`EventBus`**: The central dispatcher that routes contexts to subscribed handlers.

**Note:** Event Bus is fail-fast: if your handler raises an exception, the loop terminates immediately.

## Registering Handlers

Both `BaseTask` (and by extension `TrainTask` and `InferenceTask`) and `ModelProvider` expose a `register_events` hook. The framework calls this during configuration, passing an object containing the `EventBus`.

### Declarative Registration (Recommended)

To provide a clean developer experience, `d9d` offers a `@subscribe` decorator. Instead of manually binding each method to the event bus, you can tag your methods and use the `subscribe_annotated` helper to register them all at once.

```python
from d9d.loop.control import TrainTask, RegisterTaskEventsContext
from d9d.loop.event import (
    subscribe,
    subscribe_annotated
)
from d9d.loop.event.catalogue.train import (
    EVENT_TRAIN_MODEL_STAGES_READY,
    EVENT_TRAIN_STEP_POST,
    EventModelStagesReadyContext,
    EventStepContext,
)

class CustomTrainTask(TrainTask):
    def __init__(self):
        self._modules = []

    def register_events(self, ctx: RegisterTaskEventsContext) -> None:
        # Automatically scans this instance for @subscribe decorators
        subscribe_annotated(ctx.event_bus, self)

    @subscribe(EVENT_TRAIN_MODEL_STAGES_READY)
    def _on_model_ready(self, ctx: EventModelStagesReadyContext) -> None:
        self._modules = ctx.modules
    
    @subscribe(EVENT_TRAIN_STEP_POST)
    def _on_step_post(self, ctx: EventStepContext) -> None:
        for module in self._modules:
            print(f"Step {ctx.stepper.current_step} completed; MoE routing stats: {module.moe_stats}.")

    def compute_loss(self, ctx):
        ... # Task math overrides
```

### Manual Registration

You can also interact with the `EventBus` directly, which is useful when registering simple lambda callbacks or dynamically creating handlers.

```python
from d9d.loop.control import TrainTask, RegisterTaskEventsContext
from d9d.loop.event import subscribe
from d9d.loop.event.catalogue.train import (
    EVENT_TRAIN_OPTIMIZER_READY
)

class CustomTrainTask(TrainTask):
    def register_events(self, ctx: RegisterTaskEventsContext) -> None:
        ctx.event_bus.subscribe(
            EVENT_TRAIN_OPTIMIZER_READY, 
            lambda event_ctx: print(f"Optimizer loaded: {event_ctx.optimizer}")
        )
```

## Custom Events

You can easily define and trigger your own events inside custom logic.

```python
from d9d.loop.event import EventBus, Event
import dataclasses
from pathlib import Path

@dataclasses.dataclass(kw_only=True)
class CheckpointContext:
    step: int
    path: Path

# Define a new event
EVENT_CHECKPOINT_SAVED = Event[CheckpointContext](id="user.checkpoint_saved")

# Trigger it from somewhere in your task
def process_something(bus: EventBus):
    bus.trigger(
        EVENT_CHECKPOINT_SAVED, 
        CheckpointContext(step=1000, path=Path("/checkpoints/step_1000"))
    )
```

## API Reference

### Core Components

::: d9d.loop.event
    options:
      heading_level: 3

### Common Events

::: d9d.loop.event.catalogue.common
    options:
      heading_level: 3

### Training Events

::: d9d.loop.event.catalogue.train
    options:
      heading_level: 3

### Inference Events

::: d9d.loop.event.catalogue.inference
    options:
      heading_level: 3
