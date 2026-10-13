# Experiment Tracking

## About

The `d9d.tracker` package sends the loss, the metrics and the hyperparameters of a training job to an experiment tracker. The `trainer.logging.tracker` section of the configuration selects the tracker by its `provider` field. Only the main process logs.

## Choosing a Tracker

### Log (`log`)

Writes each scalar as one line to the Python logger of d9d, next to the other messages of d9d. It needs no extra and no options. It does not log hyperparameters.

```yaml
logging:
  period_steps: 10
  tracker:
    provider: log
```

A log line looks like this:

```text
[d9d] [local] 2026-10-04 01:08:11,755 - INFO - step 1 (stage=train): loss=15.0928
```

### Aim (`aim`)

Logs to an [Aim](https://aimstack.io/) repository. It needs the `d9d[aim]` extra and the path or URL of the repository in `repo`. By default, it also records the CPU, GPU and memory usage and the terminal output.

```yaml
logging:
  period_steps: 10
  tracker:
    provider: aim
    repo: runs/aim
```

To see the runs, start the Aim UI on the same repository:

```bash
aim up --repo runs/aim
```

### Null (`null`)

Logs nothing.

```yaml
logging:
  period_steps: 10
  tracker:
    provider: "null"  # Quoted, because YAML reads a bare null as None.
```

## What the Loop Logs

| Name | When | Value |
|:-----|:-----|:------|
| `loss` | Every step | The loss of the step, averaged over its microbatches and ranks, weighted by `loss_weight`. |
| `l2_grad_norm_total` | Every step | The total gradient norm of the step, before clipping. |
| The names of the task metrics | Every `logging.period_steps` steps and on the last step | The metrics that the task creates in `create_metrics()`, accumulated over the steps since the previous logging step. A nested name joins its keys with `/`. See [Metrics Overview](../metric/overview.md). |

Each value has the step number and the context `stage=train`.

Logging does not make the step wait for the GPU. A value waits in pinned host memory until the GPU has computed it, and reaches the tracker at a later logging call. The values can lag `logging.max_pending_steps` steps (2 by default) behind the current step. If they lag more, logging waits for the GPU to finish the oldest step. This also limits how far the host runs ahead of the GPU. Before a checkpoint and at the end of training, the loop writes all waiting values to the tracker. If training raises, the loop writes only the values that the GPU has already computed.

At the start of a run, the tracker also gets the hyperparameters: `trainer.run.hparams` from the configuration and the results of `dump_hparams()` of the task and of the model provider. Override `dump_hparams()` to log the settings of your own code, e.g. the size of the model.

## Logging Your Own Values

The context of `EVENT_TRAIN_READY` holds the `TrackerWriter` of the run. Keep it in your [event handler](../loop/interfaces/events.md) and log with `log()` and a value of one of the types in `TrackerValue` (see the [API Reference](#api-reference)). The tensors of a value can be on the GPU. Each value gets the step that is current at the call and the context `stage=train`. Only the main process writes to the tracker, so the tracker shows the values of rank 0.

This task logs a histogram of each weight matrix every 100 steps:

```python
import torch
from torch.distributed.tensor import DTensor

from d9d.loop.component import TrackerWriter
from d9d.loop.control import RegisterTaskEventsContext, TrainTask
from d9d.loop.event import subscribe, subscribe_annotated
from d9d.loop.event.catalogue.common import EventModelStagesReadyContext, EventStepContext
from d9d.loop.event.catalogue.train import (
    EVENT_TRAIN_MODEL_STAGES_READY,
    EVENT_TRAIN_READY,
    EVENT_TRAIN_STEP_POST,
    EventTrainReadyContext,
)
from d9d.tracker import Histogram

PERIOD_STEPS = 100


class WeightHistogramTask(TrainTask):
    def __init__(self):
        self._writer: TrackerWriter | None = None
        self._weights: list[tuple[str, torch.Tensor]] = []

    def register_events(self, context: RegisterTaskEventsContext) -> None:
        subscribe_annotated(context.event_bus, self)

    @subscribe(EVENT_TRAIN_MODEL_STAGES_READY)
    def _on_model_ready(self, ctx: EventModelStagesReadyContext) -> None:
        self._weights = [
            (name, param) for stage in ctx.modules for name, param in stage.named_parameters() if param.ndim == 2
        ]

    @subscribe(EVENT_TRAIN_READY)
    def _on_ready(self, ctx: EventTrainReadyContext) -> None:
        self._writer = ctx.writer

    @subscribe(EVENT_TRAIN_STEP_POST)
    def _on_step_post(self, ctx: EventStepContext) -> None:
        if ctx.schedule.current_step % PERIOD_STEPS != 0:
            return
        # EVENT_TRAIN_READY comes before the first step.
        assert self._writer is not None
        for name, weight in self._weights:
            # With FSDP, the histogram covers the shard of this rank.
            local = weight.to_local() if isinstance(weight, DTensor) else weight
            # A fixed range keeps histc on the GPU. The automatic range (min=max=0) waits for the GPU to find it.
            counts = torch.histc(local.detach().float(), bins=50, min=-0.1, max=0.1)
            self._writer.log(f"weights/{name}", Histogram(counts=counts))

    def build_forward_inputs(self, ctx):
        ...  # Task logic

    def compute_loss(self, ctx):
        ...  # Task logic
```

The histograms go to the writer and not to a [metric](../metric/overview.md). A histogram is a snapshot of the weights at one step, while a metric accumulates a value over `logging.period_steps` and synchronizes it across ranks. Only the main process writes, so with FSDP the tracker shows the shards of rank 0, and with pipeline parallelism the stage of rank 0.

A call reads a tensor on the current CUDA stream. You can change the tensor in place after the call, on the same stream, without changing the logged value. The writer works only while the run is open, until `train()` returns. Later calls raise an error.

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

Implement `BaseTrackerRun`. Its `write()` gets a `TrackerPoint` with the name, the step and the context, and a value of one of the types in `TrackerValue`. The tensors of the value are already on the CPU. End the `match` with `case _:` that raises a `ValueError`, so that a value type that your tracker does not handle yet, e.g. after an update of d9d, fails loudly. A tracker that writes only some types on purpose, such as the log tracker, can skip the others with `case _: pass`.

```python
from d9d.tracker import BaseTrackerRun, Histogram, Scalar, TrackerPoint, TrackerValue


class WandbRun(BaseTrackerRun):
    def __init__(self, run_obj):
        self._run = run_obj

    def write(self, point: TrackerPoint, value: TrackerValue):
        match value:
            case Scalar():
                self._run.log({point.name: float(value.value)}, step=point.step)
            case Histogram():
                ...  # Call self._run.log() with a wandb.Histogram built from value.counts.
            case _:
                raise ValueError(f"Tracker value type ({type(value).__name__}) is not supported by W&B.")
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

::: d9d.loop.component.TrackerWriter
