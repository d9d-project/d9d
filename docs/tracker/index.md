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

### W&B (`wandb`)

Logs to [Weights & Biases](https://wandb.ai/). It needs the `d9d[tracker-wandb]` extra, a W&B account and the project in `project`. Log in with `wandb login` or set `WANDB_API_KEY` before you start the job.

```yaml
logging:
  period_steps: 10
  tracker:
    provider: wandb
    project: my-project
    entity: my-team  # Optional. By default, the entity of the API key.
```

The context becomes a prefix of the name, e.g. `train/loss`. The hyperparameters go to the config of the run. By default, W&B does not capture the terminal output: set `capture_terminal_logs: true` to capture it.

A resumed job continues the same run. W&B drops a value whose step is lower than the last step of the run. The job may have logged some steps after its last checkpoint before it stopped. After a resume, W&B keeps the old values of these steps and drops the new ones.

With `mode: offline`, W&B writes the run to local files, which `wandb sync` uploads later. W&B does not resume a run in offline mode.

### TensorBoard (`tensorboard`)

Writes [TensorBoard](https://www.tensorflow.org/tensorboard) event files to the directory in `log_dir`. It needs the `d9d[tracker-tensorboard]` extra. It needs no account and no server.

```yaml
logging:
  period_steps: 10
  tracker:
    provider: tensorboard
    log_dir: runs/tensorboard/my-run
```

To see the run, start TensorBoard on the parent directory. It shows each subdirectory as a run:

```bash
tensorboard --logdir runs/tensorboard
```

The context becomes a prefix of the name, e.g. `train/loss`. The name, the description and the hyperparameters of the run go to the Text tab, as the entry `run`.

A resumed job writes to the same directory. TensorBoard drops the values that the stopped job logged after its last checkpoint, so the run shows each step once.

### MLflow (`mlflow`)

Logs to an [MLflow](https://mlflow.org/) tracking server or store. It needs the `d9d[tracker-mlflow]` extra and the name of the experiment in `experiment`. The tracker creates the experiment if it does not exist.

```yaml
logging:
  period_steps: 10
  tracker:
    provider: mlflow
    experiment: my-experiment
    tracking_uri: http://mlflow.example.com:5000  # Optional. By default, MLFLOW_TRACKING_URI or the MLflow default.
```

The context becomes a prefix of the name, e.g. `train/loss`. The nested hyperparameters become params with names such as `run/lr`. MLflow has no histogram type, so the tracker does not log histograms. The tracker sends the values in a background thread, so a step does not wait for the server.

A resumed job continues the same run. MLflow does not allow you to change a param, so the run keeps the hyperparameters of its first job. The job may have logged some steps after its last checkpoint before it stopped. After a resume, MLflow keeps both the old and the new values of these steps.

### Aim (`aim`)

Logs to an [Aim](https://aimstack.io/) repository. It needs the `d9d[tracker-aim]` extra and the path or URL of the repository in `repo`. By default, it also records the CPU, GPU and memory usage and the terminal output.

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

A `Histogram` holds only the counts, so W&B, TensorBoard and Aim show the bin numbers on the x-axis, not the values. MLflow and the log tracker do not log histograms.

A call reads a tensor on the current CUDA stream. You can change the tensor in place after the call, on the same stream, without changing the logged value. The writer works only while the run is open, until `train()` returns. Later calls raise an error.

## Adding a New Tracker

The configuration union and the tracker factory live in `d9d/tracker/factory.py`, so a new tracker is a change to d9d itself. Open a pull request with it. The examples below add [Comet](https://www.comet.com/).

### The Configuration

Create a Pydantic model for the tracker settings. It must contain a `provider` literal field. This field is the discriminator of the configuration union.

```python
from typing import Literal

from pydantic import BaseModel


class CometConfig(BaseModel):
    provider: Literal["comet"] = "comet"
    project: str
```

### The Run

Implement `BaseTrackerRun`. Its `write()` gets a `TrackerPoint` with the name, the step and the context, and a value of one of the types in `TrackerValue`. The tensors of the value are already on the CPU. End the `match` with `case _:` that raises a `ValueError`, so that a value type that your tracker does not handle yet, e.g. after an update of d9d, fails loudly. A tracker that writes only some types on purpose, such as the log tracker, can skip the others with `case _: pass`.

If the tracker has no tags, build the name of the series with `series_key()` from `d9d.tracker.provider.flat`. It puts the context before the name, e.g. `train/loss`, as the W&B, TensorBoard and MLflow trackers do.

```python
from d9d.tracker import BaseTrackerRun, Scalar, TrackerPoint, TrackerValue
from d9d.tracker.provider.flat import series_key


class CometRun(BaseTrackerRun):
    def __init__(self, experiment):
        self._experiment = experiment

    def write(self, point: TrackerPoint, value: TrackerValue):
        match value:
            case Scalar():
                self._experiment.log_metric(series_key(point), float(value.value), step=point.step)
            case _:
                raise ValueError(f"Tracker value type ({type(value).__name__}) is not supported by Comet.")
```

### The Tracker

Implement `BaseTracker`. It opens runs and saves the state needed to resume them.

```python
from contextlib import contextmanager

from d9d.tracker import BaseTracker, RunConfig


class CometTracker(BaseTracker[CometConfig]):
    def __init__(self, config: CometConfig):
        self._config = config
        self._experiment_key = None

    @classmethod
    def from_config(cls, config: CometConfig):
        return cls(config)

    # The checkpoint saves this state, so a resumed job continues the same run.
    def state_dict(self):
        return {"experiment_key": self._experiment_key}

    def load_state_dict(self, state_dict):
        self._experiment_key = state_dict["experiment_key"]

    @contextmanager
    def open(self, properties: RunConfig):
        # Start the experiment with the saved key, or a new one, and save its key.
        # Log properties.hparams, yield a CometRun and end the experiment.
        ...
```

### Registration

Add the configuration to the `AnyTrackerConfig` union in `d9d/tracker/factory.py`, and a `case` for it to `tracker_from_config()`. If the SDK is an optional dependency, add the extra `tracker-<provider>` to `pyproject.toml`. Import the tracker inside the `case`, so that d9d imports without the SDK and does not spend time to import an SDK that the job does not use:

```python
        case CometConfig():
            with _extra_import(config.provider):
                from .provider.comet.tracker import CometTracker  # noqa: PLC0415
            return CometTracker.from_config(config)
```

## API Reference

::: d9d.tracker

::: d9d.loop.component.TrackerWriter
