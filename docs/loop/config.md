# Configuration

## About

The `d9d.loop.config` package defines the job configuration as Pydantic models. `TrainerConfig` configures a training job, and `InferenceConfig` configures an inference job. Pydantic validates the whole configuration when you load it.

## Usage

Load the configuration from JSON and pass it to the configurator as `parameters`:

```python
from pathlib import Path

from d9d.loop.config import TrainerConfig

config = TrainerConfig.model_validate_json(Path("trainer.json").read_text(encoding="utf-8"))
```

Periodic actions, such as `checkpointing.period_steps`, take a `StepActionPeriod`. It is a step interval (e.g. `100`), `"last_step"` or `"disable"`.

## API Reference

### Main Config

::: d9d.loop.config.TrainerConfig
    options:
      heading_level: 4

::: d9d.loop.config.InferenceConfig
    options:
      heading_level: 4

### Diagnostics and Reproducibility

::: d9d.tracker.RunConfig
    options:
      heading_level: 4

::: d9d.loop.config.JobLoggerConfig
    options:
      heading_level: 4

::: d9d.loop.config.ProfilingConfig
    options:
      heading_level: 4

::: d9d.loop.config.DeterminismConfig
    options:
      heading_level: 4

### Experiment Trackers

::: d9d.tracker.AnyTrackerConfig
    options:
      heading_level: 4

::: d9d.tracker.provider.null.NullTrackerConfig
    options:
      heading_level: 4

::: d9d.tracker.provider.aim.config.AimConfig
    options:
      heading_level: 4

### Scheduling

::: d9d.loop.config.JobScheduleConfig
    options:
      heading_level: 4

### Data Prefetching

::: d9d.loop.config.DataPrefetchConfig
    options:
      heading_level: 4

### Checkpointing

::: d9d.loop.config.CheckpointingConfig
    options:
      heading_level: 4

### Model Initialization

::: d9d.loop.config.ModelStageFactoryConfig
    options:
      heading_level: 4

### Optimization

::: d9d.loop.config.GradientClippingConfig
    options:
      heading_level: 4

::: d9d.loop.config.GradientManagerConfig
    options:
      heading_level: 4

### Infrastructure

::: d9d.loop.config.PipeliningConfig
    options:
      heading_level: 4

::: d9d.loop.config.GarbageCollectionConfig
    options:
      heading_level: 4

::: d9d.loop.config.TimeoutConfig
    options:
      heading_level: 4

### Types

::: d9d.loop.config.StepActionPeriod
    options:
      heading_level: 4

::: d9d.loop.config.StepActionSpecial
    options:
      heading_level: 4
