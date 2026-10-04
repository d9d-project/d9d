# Experiment Tracking

## About

The `d9d.tracker` package gives one configuration-driven interface for logging metrics, hyperparameters and distributions during training. A common API hides the backend, such as [Aim](https://aimstack.io/), the Python logger or the null tracker that logs nothing. Each backend has a `pydantic` configuration, so you can switch backends in the configuration without changing the training loop.

!!! warning "Internal API"
    If you use the standard d9d training loop, you do not need to call this package. d9d sets up tracking based on its configuration. This page is for users who extend d9d.

## Built-in Trackers

The `logging.tracker` section of the job configuration selects the tracker by its `provider` field. Only the main process logs.

*   **`log`**: Writes each scalar as one line to the Python logger of d9d, next to the other messages of d9d. The loop logs the loss every step and the metrics every `logging.period_ste`ps` steps. It needs no extra.
*   **`aim`**: Logs to [Aim](https://aimstack.io/). It needs the `d9d[aim]` extra.
*   **`null`**: Logs nothing.

```json
"logging": {
  "period_steps": 10,
  "tracker": {"provider": "log"}
}
```

A log line looks like this. A step that logs metrics writes one more line for each metric.

```text
[d9d] [local] 2026-10-04 01:08:11,755 - INFO - step 1 (stage=train): loss=15.0928
```

## Trackers and Runs

The package splits tracking into two objects:

1.  **The tracker**: `BaseTracker`. It lives for the whole application. It holds the configuration (where to save logs) and the state (the ID of the current run). It opens runs.
2.  **The run**: `BaseTrackerRun`. It is a context-managed object, active only during the training loop. It provides `set_step`, `set_context`, `scalar` and `bins`.

The `tracker_from_config` function creates a `BaseTracker` from its Pydantic configuration.

## Resuming a Run

The tracker is **stateful**: it implements the PyTorch `Stateful` protocol. When an interrupted job resumes from a checkpoint, the tracker reattaches to the existing run instead of starting a new one.

## Adding a New Tracker

To support a new logging backend, such as Weights & Biases or MLflow, implement three components and register them in the factory.

### The Configuration

Create a Pydantic model for the tracker settings. It must contain a `provider` literal field. This field is the discriminator of the configuration union.

```python
from typing import Literal
from pydantic import BaseModel

class WandbConfig(BaseModel):
    provider: Literal["wandb"] = "wandb"
    project: str
    entity: str | None = None
```

### The Run

Implement `BaseTrackerRun`. It maps the d9d calls (`scalar`, `bins`, ...) to the calls of your backend SDK.

```python
from d9d.tracker import BaseTrackerRun

class WandbRun(BaseTrackerRun):
    def __init__(self, run_obj):
        self._run = run_obj
        self._step = 0

    def set_step(self, step: int):
        self._step = step

    # ... Implement set_context(), scalar() and bins() to call self._run.log() ...
```

### The Tracker

Implement `BaseTracker`. It opens runs and saves the state needed to resume them.

```python
from contextlib import contextmanager
from d9d.tracker import BaseTracker, RunConfig

class WandbTracker(BaseTracker[WandbConfig]):
    def __init__(self, config: WandbConfig):
        self._config = config
        self._run_id = None  # State to save

    @classmethod
    def from_config(cls, config: WandbConfig):
        return cls(config)

    def state_dict(self):
        # Saved to the checkpoint.
        return {"run_id": self._run_id}

    def load_state_dict(self, state_dict):
        # Restored from the checkpoint.
        self._run_id = state_dict.get("run_id")

    @contextmanager
    def open(self, properties: RunConfig):
        # Start the run, e.g. wandb.init(id=self._run_id, resume="allow", ...)
        # self._run_id = ...
        # yield WandbRun(...)
        # Finish the run.
        ...
```

### Registration

To make `tracker_from_config` recognize the new tracker, edit `d9d/tracker/factory.py`.

Add your configuration to the `AnyTrackerConfig` type alias:

```python
AnyTrackerConfig = Annotated[
    AimConfig | NullTrackerConfig | WandbConfig,  # <--- Add here
    Field(discriminator="provider")
]
```

Register the mapping in `_MAP`. If the SDK is an optional dependency, wrap the import in `try`/`except`:

```python
try:
    from .provider.wandb.tracker import WandbTracker
    _MAP[WandbConfig] = WandbTracker
except ImportError as e:
    _MAP[WandbConfig] = _TrackerImportFailed(dependency="wandb", exception=e)
```

## API Reference

::: d9d.tracker
