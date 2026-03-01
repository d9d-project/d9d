---
title: Experiment Tracking
---

# Experiment Tracking

## About

!!! warning "Warning:" 
    If you are utilizing the standard `d9d` training infrastructure, you **do not** need to call these functions manually. The framework automatically handles tracking based on configuration. This package is primarily intended for users extending `d9d`.

The `d9d.tracker` package provides a unified, configuration-driven interface for logging metrics, hyperparameters, and distributions during training. 

It abstracts the specific backend (such as [Aim](https://aimstack.io/) or simple console logging) behind a common API. This coupled with a `pydantic` configuration system allows users to switch logging backends via configuration files without changing a single line of training loop code.

Crucially, the tracker is **State Aware**. It implements the PyTorch `Stateful` protocol, ensuring that if a training job is interrupted and resumed, the tracker automatically re-attaches to the existing experiment run rather than creating a fragmented new one.

## Architecture Separation of Concerns

The module splits tracking logic into two distinct phases:

1.  **The Tracker (Factory/Manager)**: Represented by `BaseTracker`. This object persists throughout the lifecycle of the application. It holds configuration (where to save logs) and state (the ID of the current run). It is responsible for creating "Runs".
2.  **The Run (Session)**: Represented by `BaseTrackerRun`. This is a context-managed object active only during the actual training loop. It handles the `set_step`, `scalar`, and `bins` operations.

There is also factory method called `tracker_from_config` that can create a `BaseTracker` object based on Pydantic configuration.

## Adding a New Tracker

To support a new logging backend (e.g., Weights & Biases, MLFlow), you need to implement three components and register them in the factory.

### The Configuration

Create a Pydantic model for your tracker's settings. Functionally, it must contain a `provider` literal field which acts as the discriminator for the polymorphic deserialization.

```python
from typing import Literal
from pydantic import BaseModel

class WandbConfig(BaseModel):
    provider: Literal['wandb'] = 'wandb'
    project: str
    entity: str | None = None
```

### The Run Handler
Implement `BaseTrackerRun`. This class maps `d9d` calls (`scalar`, `bins`) to the specific calls of your backend SDK.

```python
from d9d.tracker import BaseTrackerRun

class WandbRun(BaseTrackerRun):
    def __init__(self, run_obj):
        self._run = run_obj
        self._step = 0
        
    def set_step(self, step: int):
        self._step = step
    
    # ... implement scalar(), bins(), etc. to call self._run.log()
```

### The Tracker Factory
Implement `BaseTracker`. This handles initialization and state persistence (resuming).

```python
from contextlib import contextmanager
from d9d.tracker import BaseTracker, RunConfig

class WandbTracker(BaseTracker[WandbConfig]):
    def __init__(self, config: WandbConfig):
        self.config = config
        self.run_id = None # State to persist
        
    def state_dict(self):
        # This is saved to the checkpoint
        return {"run_id": self.run_id}

    def load_state_dict(self, state_dict):
        # This is restored from the checkpoint
        self.run_id = state_dict.get("run_id")
        
    @contextmanager
    def open(self, props: RunConfig):
        # Logic to init e.g. wandb.init(id=self.run_id, resume="allow", ...)
        # self.run_id = ...
        # yield WandbRun(...)
        # cleanup if necessary
```

### Registration
To make `tracker_from_config` recognize your new tracker, you must modify `d9d/tracker/factory.py`.

Add your config to `AnyTrackerConfig` type alias:

```python
AnyTrackerConfig = Annotated[
    AimConfig | NullTrackerConfig | WandbConfig, # <--- Add here
    Field(discriminator='provider')
]
```

Register the mapping in `_MAP` (wrapping imports in try/except is recommended if the SDK is an optional dependency):
```python
try:
    from .provider.wandb.tracker import WandbTracker
    _MAP[WandbConfig] = WandbTracker
except ImportError as e:
    _MAP[WandbConfig] = _TrackerImportFailed('wandb', e)
```

::: d9d.tracker
    options:
        show_root_heading: true
        show_root_full_path: true
