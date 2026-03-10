# Learning Rate Scheduler

## Auto Scheduler

For standard PyTorch usage, `d9d` includes the `d9d.loop.auto` package. These providers ingest a Pydantic configuration object and manage the creation of standard schedulers.

Supports [Piecewise Linear](../lr_scheduler/piecewise.md) schedules (warmup, hold, decay).

```python
from d9d.loop.auto import AutoLRSchedulerProvider, AutoLRSchedulerConfig

cfg = """
{
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
"""

provider = AutoLRSchedulerProvider(
    AutoLRSchedulerConfig.model_validate_json(cfg)
)
```

::: d9d.loop.auto.auto_lr_scheduler

## Interface

If you need a custom learning rate scheduler, you implement the `LRSchedulerProvider` protocol.

::: d9d.loop.control.lr_scheduler_provider
