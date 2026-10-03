from collections.abc import Callable

from torch import nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import LRScheduler

SchedulerFactory = Callable[[Optimizer], LRScheduler]


def _get_history(factory: SchedulerFactory, num_steps: int, init_lr: float) -> list[float]:
    optimizer = SGD(nn.Linear(1, 1).parameters(), lr=init_lr)

    scheduler = factory(optimizer)

    lrs = []

    for _ in range(num_steps):
        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        scheduler.step()

    return lrs


def visualize_lr_scheduler(factory: SchedulerFactory, num_steps: int, init_lr: float = 1.0):
    """Plots a learning rate schedule as an interactive Plotly chart.

    The function builds the scheduler for a dummy optimizer, steps it ``num_steps`` times and plots the
    learning rate at each step.

    Args:
        factory: Callable that builds the scheduler for a given optimizer.
        num_steps: Number of steps to simulate.
        init_lr: Initial learning rate of the dummy optimizer.

    Raises:
        ImportError: If ``plotly`` is not installed.
    """
    try:
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("Scheduler visualization requires plotly. Install the d9d[visualization] extra.") from e
    lrs = _get_history(factory, num_steps, init_lr)
    steps = list(range(num_steps))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=lrs,
            mode="lines",
            name="Learning Rate",
            line={"color": "#636EFA", "width": 3},
            hovertemplate="<b>Step:</b> %{x}<br><b>LR:</b> %{y:.6f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": "Scheduler", "y": 0.95, "x": 0.5, "xanchor": "center", "yanchor": "top"},
        xaxis_title="Steps",
        yaxis_title="Learning Rate",
        template="plotly_white",
        hovermode="x unified",
        height=500,
    )

    fig.show()
