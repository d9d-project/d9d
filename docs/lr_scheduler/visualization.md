# Schedule Visualization

## About

`visualize_lr_scheduler` plots a learning rate schedule as an interactive [Plotly](https://plotly.com/python/) chart. It helps you check a multiphase schedule before you train with it. It needs the `d9d[visualization]` extra.

## Usage

Pass a function that builds your scheduler for a given optimizer. `visualize_lr_scheduler` builds the scheduler for a dummy optimizer, steps it `num_steps` times and plots the learning rate at each step.

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


# Opens an interactive plot in the browser or notebook.
visualize_lr_scheduler(
    factory=create_scheduler,
    num_steps=100,  # Number of steps to simulate
    init_lr=1e-3  # Base learning rate of the dummy optimizer
)
```

## API Reference

::: d9d.lr_scheduler.visualizer
