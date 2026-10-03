# Piecewise Scheduler

## About

The `d9d.lr_scheduler.piecewise` module builds piecewise learning rate schedules, such as "warmup, hold, decay". You chain phases together instead of writing an `LRScheduler` subclass or a `LambdaLR` function for each schedule.

## Phases

A schedule computes a multiplier for the base learning rate of the optimizer. It starts at `initial_multiplier`. Each phase moves the multiplier from the end value of the previous phase to its own `target_multiplier`, along a curve.

A phase can last:

*   a fixed number of steps (`for_steps`, or `"mode": "steps"` in a config);
*   until a fraction of the total steps (`until_percentage`, or `"mode": "percentage"`);
*   until the end of training (`fill_rest`, or `"mode": "rest"`).

The last two need `total_steps`. After the last phase, the multiplier stays at its end value.

## Available Curves

The following curves interpolate the multiplier within a phase:

| Curve Class        | Curve Config    | Description                                                                 |
|:-------------------|-----------------|:----------------------------------------------------------------------------|
| `CurveLinear`      | `"linear"`      | Straight-line interpolation.                                                |
| `CurveCosine`      | `"cosine"`      | Half-period cosine interpolation (cosine annealing).                        |
| `CurvePoly(power)` | `"poly"`        | Polynomial interpolation. `power=1` is linear, `power=2` is quadratic.      |
| `CurveExponential` | `"exponential"` | Exponential interpolation, linear in log space. Values below `1e-8` are treated as `1e-8`. |

## Usage

### Python API

This example builds a "linear warmup, hold, cosine decay" schedule:

```python
import torch
from d9d.lr_scheduler.piecewise import CurveCosine, CurveLinear, piecewise_schedule

optimizer: torch.optim.Optimizer = ...
total_steps: int = 1000

# 1. Start at 0.0.
# 2. Warm up linearly to 1.0 over 100 steps.
# 3. Hold at 1.0 until 50% of the training steps.
# 4. Decay along a cosine to 0.1 for the rest of training.
scheduler = (
    piecewise_schedule(initial_multiplier=0.0, total_steps=total_steps)
    .for_steps(100, target_multiplier=1.0, curve=CurveLinear())
    .until_percentage(0.5, target_multiplier=1.0, curve=CurveLinear())
    .fill_rest(target_multiplier=0.1, curve=CurveCosine())
    .build(optimizer)
)
```

### Pydantic API

```python
import torch
from d9d.lr_scheduler.piecewise import PiecewiseSchedulerConfig, piecewise_scheduler_from_config

optimizer: torch.optim.Optimizer = ...
total_steps: int = 1000

raw_config_json = """
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

scheduler_config = PiecewiseSchedulerConfig.model_validate_json(raw_config_json)

scheduler = piecewise_scheduler_from_config(
    config=scheduler_config,
    optimizer=optimizer,
    total_steps=total_steps
)
```

## API Reference

::: d9d.lr_scheduler.piecewise
