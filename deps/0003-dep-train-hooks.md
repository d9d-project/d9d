---
DEP: 0003
Title: Event Bus Training Hooks
Author: Daniil Sergeev @DaniilSergeev17 & Maksim Afanasyev @mrapplexz
Status: Draft
Type: Feature
Created: 2026-03-12
---

# DEP-0003: Event Bus Training Hooks

## Abstract

This proposal introduces a typed **Event Bus** as the extension mechanism for d9d's training and inference loops.
Instead of a fixed protocol with hardcoded lifecycle methods, the framework defines a set of **typed events** (
`Event[TContext]`) and a lightweight **publish-subscribe bus** (`EventBus`). Both loops trigger events at well-defined
points; any user logic component — `ModelProvider`, `BaseTask` — subscribes handlers to the events it cares about.

This decouples the framework from model-specific logic entirely. Components subscribe only to the events they need, new
events can be introduced without changing any interface, and multiple independent subscribers coexist naturally.

## Motivation

### Current State

The `Trainer.train()` loop in d9d executes a fixed sequence of operations per optimizer step:

```python
for each batch_group:
    for each micro - batch:
        forward_backward()
    gradient_manager.sync_and_scale()
    gradient_clipper.clip_and_log()
    optimizer.step()
    lr_scheduler.step()
    gradient_manager.zero_grad()
```

This loop has no extension points. Model-specific logic that must run at specific moments (e.g., resetting counters
before micro-batches, synchronizing statistics before the optimizer step, applying non-gradient updates after the
optimizer step) must either be hardcoded into the framework, hacked into `TrainTask.compute_loss()` at the wrong time,
or done outside the framework entirely.

The inference loop has the same problem — no way to inject per-step logic without modifying framework code.

### Why an Event Bus

1. **No empty stubs.** A fixed hook protocol (e.g., `on_step_start`, `on_pre_optimizer`, `on_post_optimizer`) forces
   every implementor to define all methods even when it only needs one. With an event bus, a component subscribes only
   to the events it uses.

2. **Open for extension & composition.** Adding a new lifecycle point (e.g., `POST_BACKWARD`, `CHECKPOINT_SAVED`) means
   defining a new `Event` constant and a single `trigger()` call in the loop. No existing interfaces, protocols, or
   subscribers change. Multiple components can subscribe to the same event independently. An MoE load balancing manager,
   an EMA updater, and a custom logger all coexist without knowing about each other — they are separate subscribers, not
   methods on the same object.

## Design Proposal

### `Event[TContext]` Descriptor

An event is a lightweight, immutable, typed descriptor. The generic parameter `TContext` carries the type of context
object that handlers receive.

```python
import dataclasses
from typing import Generic, TypeVar

TContext = TypeVar("TContext")


@dataclasses.dataclass(frozen=True)
class Event(Generic[TContext]):
    id: str
    ...
```

Events are defined as module-level constants. Static typing is enforced by annotating the variable with the generic
parameter:

```python
EVENT_STEP_START: Event[TrainingStepContext] = Event(id="training.step_start")
```

Anyone can define new events in their own modules. The framework has no closed set of events.

### `EventBus`

The event bus is a minimal publish-subscribe dispatcher. It uses the `Event` objects themselves as dictionary keys,
avoiding string lookups.

```python
# d9d/loop/event/core.py

from typing import Callable, Any


class EventBus:
    def subscribe(
            self,
            event: Event[TContext],
            handler: Callable[[TContext], None]
    ) -> None:
        ...

    def trigger(
            self,
            event: Event[TContext],
            context: TContext
    ) -> None:
        ...
```

**Fail-Fast Error Handling:** Event dispatch is strictly synchronous and fail-fast. If any handler raises an exception
during `trigger()`, the iteration stops immediately. The `EventBus` does not catch or aggregate exceptions. This is
intentional: failing fast ensures that a corrupted operation (e.g., a failed parameter sync or checkpointing error)
stops the process rather than allowing the loop to silently continue in an invalid state.

### Registration Entry Points

#### `ModelProvider.register_events()`

`ModelProvider` has a non-abstract `register_events()` method with a default no-op. Existing providers are unaffected.

#### `BaseTask.register_events()`

`BaseTask` has `register_events()` with a default no-op. Both `TrainTask` and `InferenceTask` inherit it, so any task
can subscribe to events.

### Declarative Registration (`@subscribe`)

To provide better developer experience, d9d provides a declarative top-level `@subscribe` decorator.

The `@subscribe` decorator does not register the event immediately. Instead, it statically tags the underlying function
with a target event using a private metadata marker.

To finalize registration, a user passes the object instance and the event bus to the top-level `subscribe_annotated`
utility. This utility uses Python's `inspect.getmembers()` to scan the instantiated class for tagged methods and
correctly binds the instance (`self`) to the handlers.

## Usage

### Basic Example

Here is an example of a custom training task that prints MoE stats at the end of every step.
It uses the event bus to capture the model stages during setup, and then hooks into the post-step phase during
execution.

```python
from d9d.loop.control import TrainTask, RegisterTaskEventsContext
from d9d.loop.event import (
    EVENT_TRAIN_MODEL_STAGES_READY,
    EVENT_TRAIN_STEP_POST,
    EventModelStagesReadyContext,
    EventStepContext
)


class CustomTrainTask(TrainTask):
    def __init__(self):
        # We don't have the model at init time. We will get it via the EventBus!
        self._modules = None

    def register_events(self, ctx: RegisterTaskEventsContext) -> None:
        # 1. Subscribe to DI config events to steal references to built objects
        ctx.event_bus.subscribe(EVENT_TRAIN_MODEL_STAGES_READY, self._on_model_ready)

        # 2. Subscribe to runtime events to execute logic safely
        ctx.event_bus.subscribe(EVENT_TRAIN_STEP_POST, self._on_step_post)

    def _on_model_ready(self, event_ctx: EventModelStagesReadyContext) -> None:
        self._modules = event_ctx.modules

    def _on_step_post(self, event_ctx: EventStepContext) -> None:
        for module in self._modules:
            print(module.tokens_per_layer())

    def compute_loss(self, ctx):
        ...  # Required TrainTask math overrides 
```

### Custom Events

Users can define their own events and trigger them from custom code:

```python
from d9d.loop.event.core import Event

import dataclasses
from pathlib import Path


@dataclasses.dataclass(kw_only=True)
class CheckpointContext:
    step: int
    path: Path


EVENT_CHECKPOINT_SAVED: Event[CheckpointContext] = Event(id="user.checkpoint_saved")

event_bus.subscribe(EVENT_CHECKPOINT_SAVED, lambda ctx: upload_to_s3(ctx.path))

event_bus.trigger(EVENT_CHECKPOINT_SAVED, CheckpointContext(step=1000, path=Path("/checkpoints/step_1000")))
```

### Declarative Registration

Users can cleanly group their lifecycle logic using `@subscribe`. Registration becomes a single line inside the `register_events` hook using the `subscribe_annotated` helper.

```python
from d9d.loop.control import TrainTask, RegisterTaskEventsContext
from d9d.loop.event import (
    EVENT_TRAIN_MODEL_STAGES_READY,
    EVENT_TRAIN_STEP_POST,
    EventModelStagesReadyContext,
    EventStepContext,
    subscribe,
    subscribe_annotated
)


class LoggingHandler:
    def __init__(self):
        self._modules = []
    
    @subscribe
    def on_model_ready(self, ctx: EventModelStagesReadyContext) -> None:
        self._modules = ctx.modules
    
    @subscribe
    def on_step_post(self, ctx: EventStepContext) -> None:
        for module in self._modules:
            print(module.tokens_per_layer())


class CustomTrainTask(TrainTask):
    def register_events(self, ctx: RegisterTaskEventsContext) -> None:
        subscribe_annotated(ctx.event_bus, LoggingHandler())

    def compute_loss(self, ctx):
        ...  # Required TrainTask math overrides
```

## Backward Compatibility

Fully backward compatible. `ModelProvider.register_events()` and `BaseTask.register_events()` have default no-op
implementations. Existing subclasses that do not override these methods are unaffected — no handlers are registered, and
`trigger()` calls iterate over empty lists.

Both `TrainJobState` and `InferenceJobState` gain a new `event_bus` field, which is always present (created by the
configurator). No existing fields are removed or renamed.

## Alternatives Considered

### Fixed `TrainingHook` Protocol

Let's consider `TrainingHook` protocol with fixed methods (i.e. `on_step_start`, `on_pre_optimizer`,
`on_post_optimizer`). `ModelProvider` returned a list of `TrainingHook` instances, and the `Trainer` iterated over them
at each lifecycle point.

This approach works but has drawbacks:

- Every hook must implement all three methods, even if it only needs one (e.g., EMA only needs
  `on_post_optimizer`). Default no-op implementations mitigate this but add boilerplate.
- Adding a new lifecycle point (e.g., `on_post_backward`) requires changing the protocol — users
  cannot create their own events.

The event bus solves all of these: subscribe only to what you need, add new events without changing existing code, and
register from any component through the same `EventBus`.

### Callback Lists

An alternative is to store per-event callback lists directly on `TrainJobState` (e.g.,
`on_step_start_callbacks: list[Callable]`). This avoids the `EventBus` abstraction but loses the typed event
descriptors, makes adding new events require changing the state class, and scatters registration logic across multiple
fields. The `EventBus` centralizes all subscriptions in one object with a uniform API.
