# Experiment Tracking

## About

The `d9d.tracker` package sends the loss, the metrics and the hyperparameters of a training job to an experiment tracker. The `trainer.logging.tracker` section of the configuration selects the tracker by its `provider` field. Only the main process logs.

## Choosing a Tracker

### Log (`log`)

Writes each scalar as one line to the Python logger of d9d, next to the other messages of d9d. It needs no extra and no options. It does not log hyperparameters.

```json
"logging": {
  "period_steps": 10,
  "tracker": {"provider": "log"}
}
```

A log line looks like this:

```text
[d9d] [local] 2026-10-04 01:08:11,755 - INFO - step 1 (stage=train): loss=15.0928
```

### Aim (`aim`)

Logs to an [Aim](https://aimstack.io/) repository. It needs the `d9d[aim]` extra and the path or URL of the repository in `repo`. By default, it also records the CPU, GPU and memory usage and the terminal output.

```json
"logging": {
  "period_steps": 10,
  "tracker": {"provider": "aim", "repo": "runs/aim"}
}
```

To see the runs, start the Aim UI on the same repository:

```bash
aim up --repo runs/aim
```

### Null (`null`)

Logs nothing.

```json
"logging": {
  "period_steps": 10,
  "tracker": {"provider": "null"}
}
```

## What the Loop Logs

| Name | When | Value |
|:-----|:-----|:------|
| `loss` | Every step | The loss of the step, averaged over all microbatches and ranks. |
| The names of the task metrics | Every `logging.period_steps` steps and on the last step | The metrics that the task creates in `create_metrics()`. A nested name joins its keys with `/`. See [Metrics Overview](../metric/overview.md). |
| `l2_grad_norm_total` | Every `gradient_clipping.log_total_steps` steps | The total gradient norm before clipping. |

Each value has the step number and the context `stage=train`.

At the start of a run, the tracker also gets the hyperparameters: `trainer.run.hparams` from the configuration and the results of `dump_hparams()` of the task and of the model provider. Override `dump_hparams()` to log the settings of your own code, e.g. the size of the model.

## Adding a New Tracker

The configuration union and the tracker factory live in `d9d/tracker/factory.py`, so a new tracker is a change to d9d itself. Open a pull request with it.

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

Implement `BaseTrackerRun`. It maps `set_step()`, `set_context()`, `scalar()` and `bins()` to the calls of your backend SDK.

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
        self._run_id = None

    @classmethod
    def from_config(cls, config: WandbConfig):
        return cls(config)

    # The checkpoint saves this state, so a resumed job continues the same run.
    def state_dict(self):
        return {"run_id": self._run_id}

    def load_state_dict(self, state_dict):
        self._run_id = state_dict["run_id"]

    @contextmanager
    def open(self, properties: RunConfig):
        # Start the run, e.g. wandb.init(id=self._run_id, resume="allow", ...), and save its ID.
        # Log properties.hparams, yield a WandbRun and finish the run.
        ...
```

### Registration

Add the configuration to `AnyTrackerConfig` in `d9d/tracker/factory.py`:

```python
AnyTrackerConfig = Annotated[
    AimConfig | LogTrackerConfig | NullTrackerConfig | WandbConfig,
    Field(discriminator="provider"),
]
```

Then map the configuration to the tracker in `_MAP`. If the SDK is an optional dependency, import it in `try`/`except`, so that d9d still imports without it:

```python
try:
    from .provider.wandb.tracker import WandbTracker

    _MAP[WandbConfig] = WandbTracker
except ImportError as e:
    _MAP[WandbConfig] = _TrackerImportFailed(dependency="wandb", exception=e)
```

## API Reference

::: d9d.tracker
