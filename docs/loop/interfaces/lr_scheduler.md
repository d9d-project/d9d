# Learning Rate Scheduler

## About

The loop creates one learning rate scheduler per optimizer through an `LRSchedulerProvider`. You can configure a standard schedule with `AutoLRSchedulerProvider`, or implement the `LRSchedulerProvider` protocol yourself.

## Auto Scheduler

The `d9d.loop.auto` package builds schedulers from a Pydantic configuration. It supports [piecewise](../../lr_scheduler/piecewise.md) schedules (warmup, hold, decay). The `name` field selects the scheduler.

## Custom Scheduler

For a custom scheduler, implement the `LRSchedulerProvider` protocol. It receives the optimizer and the total number of steps, and returns the scheduler.

## Usage

`AutoLRSchedulerConfig` is a discriminated union, so validate it with a Pydantic `TypeAdapter`:

```python
from pydantic import TypeAdapter

from d9d.loop.auto import AutoLRSchedulerConfig, AutoLRSchedulerProvider

cfg = """
{
    "name": "piecewise",
    "scheduler": {
        "initial_multiplier": 0.0,
        "phases": [
            {
                "mode": "steps",
                "steps": 100,
                "target_multiplier": 1.0,
                "curve": { "type": "linear" }
            },
            {
                "mode": "rest",
                "target_multiplier": 0.1,
                "curve": { "type": "cosine" }
            }
        ]
    }
}
"""

provider = AutoLRSchedulerProvider(TypeAdapter(AutoLRSchedulerConfig).validate_json(cfg))
```

Inside a larger Pydantic config, declare a field of type `AutoLRSchedulerConfig` instead.

## API Reference

::: d9d.loop.auto.auto_lr_scheduler

::: d9d.loop.control.lr_scheduler_provider
