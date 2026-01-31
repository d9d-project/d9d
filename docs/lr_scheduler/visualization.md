---
title: Visualization
---

# Schedule Visualization

## About

It is often difficult to mentally visualize complex multiphase learning rate schedules.

To address this, d9d allows you to visualize the resulting learning rate curve interactively using `plotly`.

## Usage Example

The `visualize_lr_scheduler` function takes a factory function that constructs your scheduler, simulates a training run, and plots the learning rate history.

```python
import torch

from d9d.lr_scheduler.visualizer import visualize_lr_scheduler
from d9d.lr_scheduler.piecewise import piecewise_schedule, CurveLinear, CurveCosine

def create_scheduler(optimizer: torch.optim.Optimizer):
    return (
        piecewise_schedule(initial_multiplier=0.0, total_steps=100)
        .for_steps(10, 1.0, CurveLinear())
        .fill_rest(0.0, CurveCosine())
        .build(optimizer)
    )

# Opens an interactive plot in browser/notebook
visualize_lr_scheduler(
    factory=create_scheduler,
    num_steps=100,      # Duration to simulate
    init_lr=1e-3        # Base LR to visualize
)
```

## API Reference

::: d9d.lr_scheduler.visualizer
    options:
        show_root_heading: true
        show_root_full_path: true
