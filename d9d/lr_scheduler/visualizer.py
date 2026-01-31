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
    """
    Visualizes the learning rate schedule using Plotly.

    This function simulates the training process for `num_steps` to record the LR changes
    and generates an interactive plot.

    Args:
        factory: A callable that accepts an Optimizer and returns an LRScheduler.
        num_steps: The number of steps to simulate.
        init_lr: The initial learning rate to set on the dummy optimizer.

    Raises:
        ImportError: If the `plotly` library is not installed.
    """

    try:
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("You have to install `plotly` dependency to use scheduler visualization") from e
    lrs = _get_history(factory, num_steps, init_lr)
    steps = list(range(num_steps))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=steps,
        y=lrs,
        mode="lines",
        name="Learning Rate",
        line={"color": "#636EFA", "width": 3},
        hovertemplate="<b>Step:</b> %{x}<br><b>LR:</b> %{y:.6f}<extra></extra>"
    ))

    fig.update_layout(
        title={
            "text": "Scheduler",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top"
        },
        xaxis_title="Steps",
        yaxis_title="Learning Rate",
        template="plotly_white",
        hovermode="x unified",
        height=500
    )

    fig.show()
