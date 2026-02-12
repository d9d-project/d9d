---
title: Configuration
---

# Configuration Schemas

The `d9d.loop.config` package defines the structure for configuring the training job using Pydantic models. This ensures strict validation of configurations (e.g., ensuring global batch size is divisible by microbatch size and DP size).



## Main Config

### TrainerConfig

The top-level configuration object passed to `TrainingConfigurator`.

::: d9d.loop.config.TrainerConfig
    options:
      heading_level: 3


### InferenceConfig

The top-level configuration object passed to `InferenceConfigurator`.

::: d9d.loop.config.InferenceConfig
    options:
      heading_level: 3

## Sub-Configurations

### Diagnostics & Reproducibility

::: d9d.tracker.RunConfig
    options:
      heading_level: 3

::: d9d.loop.config.JobLoggerConfig
    options:
      heading_level: 3

::: d9d.loop.config.ProfilingConfig
    options:
      heading_level: 3

::: d9d.loop.config.DeterminismConfig
    options:
      heading_level: 3

### Experiment Trackers

::: d9d.tracker.AnyTrackerConfig
    options:
      heading_level: 3

::: d9d.tracker.provider.null.NullTrackerConfig
    options:
      heading_level: 3

::: d9d.tracker.provider.aim.config.AimConfig
    options:
      heading_level: 3

### Batching & Data

::: d9d.loop.config.BatchingConfig
    options:
      heading_level: 3

::: d9d.loop.config.DataLoadingConfig
    options:
      heading_level: 3

### Checkpointing

::: d9d.loop.config.CheckpointingConfig
    options:
      heading_level: 3

### Model Initialization

::: d9d.loop.config.ModelStageFactoryConfig
    options:
      heading_level: 3

### Optimization

::: d9d.loop.config.GradientClippingConfig
    options:
      heading_level: 3

::: d9d.loop.config.GradientManagerConfig
    options:
      heading_level: 3

### Infrastructure

::: d9d.loop.config.PipeliningConfig
    options:
      heading_level: 3

::: d9d.loop.config.GarbageCollectionConfig
    options:
      heading_level: 3

::: d9d.loop.config.TimeoutConfig
    options:
      heading_level: 3

## Types

::: d9d.loop.config.StepActionPeriod
    options:
      heading_level: 3

::: d9d.loop.config.StepActionSpecial
    options:
      heading_level: 3
