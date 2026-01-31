---
title: Piecewise Scheduler
---

# Piecewise Scheduler

## About

The `d9d.lr_scheduler.piecewise` module provides a flexible, builder-based system for constructing piecewise learning rate schedules.

Instead of writing custom `LRScheduler` subclasses, manual functions for `LambdaLR` for every variation of piecewise schedule (i.e. "Warmup + Hold + Decay"), you can construct any such a schedule declaratively by chaining phases together.

## Usage Example

Here is how to create a standard "Linear Warmup + Hold + Cosine Decay" schedule:

```python
import torch
from d9d.lr_scheduler.piecewise import *

optimizer: torch.optim.Optimizer = ...
total_steps: int = 1000

# Define Schedule
# 1. Start at 0.0
# 2. Linear warmup to 1.0*LR over 100 steps
# 3. Stay at 1.0 * LR until 50% of training steps
# 3. Cosine decay to 0.1 (10% of LR) for the rest of training
scheduler = (
    piecewise_schedule(initial_multiplier=0.0, total_steps=total_steps)
    .for_steps(100, target_multiplier=1.0, curve=CurveLinear())
    .until_percentage(0.5, target_multiplier=1.0, curve=CurveLinear())
    .fill_rest(target_multiplier=0.1, curve=CurveCosine())
    .build(optimizer)
)
```

## Available Curves

The following curve classes are available to interpolate values between phases:

| Curve Class        | Description                                                                 |
|:-------------------|:----------------------------------------------------------------------------|
| `CurveLinear`      | Standard straight-line interpolation.                                       |
| `CurveCosine`      | Half-period cosine interpolation (Cosine Annealing).                        |
| `CurvePoly(power)` | Polynomial interpolation. `power=1` is linear, `power=2` is quadratic, etc. |
| `CurveExponential` | Exponential (log-linear) interpolation.                                     |

## API Reference

::: d9d.lr_scheduler.piecewise
    options:
        show_root_heading: true
        show_root_full_path: true
