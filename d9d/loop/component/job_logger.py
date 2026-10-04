from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.profiler import record_function

from d9d.core import pytree
from d9d.core.dist_context import DistributedContext
from d9d.core.types import PyTree, ScalarTree
from d9d.internals.metric_collector import AsyncMetricCollector
from d9d.internals.state import load_state_dict_main_process, state_dict_main_process
from d9d.loop.config import JobLoggerConfig
from d9d.metric.impl.container import ComposeMetric
from d9d.tracker import BaseTracker, BaseTrackerRun, RunConfig, tracker_from_config
from d9d.tracker.provider.null import NullTrackerConfig

from .job_schedule import JobSchedule


def _flatten_pytree_for_metrics(tree: PyTree[float]) -> dict[str, float]:
    flat_dict = {}

    for path_tuple, value in pytree.tree_leaves_with_path(tree):
        flat_key = "/".join(str(segment) for segment in path_tuple)
        flat_dict[flat_key] = value

    return flat_dict


class JobLogger(Stateful):
    """Logger that sends the loss, the gradient norm and the metrics of a job to the experiment tracker.

    The loss and the gradient norm are logged every step. Metrics are aggregated across ranks and logged periodically.
    """

    def __init__(
        self,
        dist_context: DistributedContext,
        config: JobLoggerConfig,
        metrics: ComposeMetric,
        schedule: JobSchedule,
        run_config: RunConfig,
        additional_hparams: ScalarTree,
    ):
        """Constructs the ``JobLogger`` object.

        Args:
            dist_context: The distributed context.
            config: The logging configuration.
            metrics: The metrics to compute and log.
            schedule: The job schedule that tracks the current step.
            run_config: The tracker run configuration.
            additional_hparams: Additional hyperparameters to log for this run.
        """
        self._dist_context = dist_context
        self._config = config
        self._schedule = schedule
        self._run_config = run_config.model_copy(
            deep=True, update={"hparams": {"run": run_config.hparams, "params": additional_hparams}}
        )

        self._tracker = self._build_tracker()
        self._metric_collector = AsyncMetricCollector(metrics)

    def _build_tracker(self) -> BaseTracker:
        if self._dist_context.is_main_process:
            return tracker_from_config(self._config.tracker)
        else:
            return tracker_from_config(NullTrackerConfig())

    @contextmanager
    def new_run(self) -> Generator[BaseTrackerRun, None, None]:
        """Opens a new experiment run.

        Yields:
            The active tracker run interface.
        """
        with self._tracker.open(self._run_config) as run:
            yield run

    @contextmanager
    def install(self):
        """Binds the async metric collector to the device while the context is open."""
        self._metric_collector.bind()
        yield
        self._metric_collector.unbind()

    def trigger_sync(self):
        """Starts the async aggregation of metrics across ranks if this step logs metrics.

        The communication overlaps with other work until ``log`` is called.
        """
        if not self._schedule.should_do_action(self._config.period_steps, enable_on_last_step_if_periodic=True):
            return

        self._metric_collector.schedule_collection(self._dist_context)

    def log(self, run: BaseTrackerRun, loss: torch.Tensor, grad_norm: torch.Tensor):
        """Logs the loss and the gradient norm of the current step and, on logging steps, the metric results.

        The metric results are the ones started by ``trigger_sync``.

        Args:
            run: The active tracker run.
            loss: The global loss of the current step, as a scalar tensor.
            grad_norm: The global gradient norm of the current step, as a scalar tensor.
        """
        with record_function("Logging"):
            # One copy to the host for both values.
            loss_value, grad_norm_value = torch.stack([loss.float(), grad_norm.float()]).tolist()
            run.scalar("loss", loss_value)
            run.scalar("l2_grad_norm_total", grad_norm_value)

            if not self._schedule.should_do_action(self._config.period_steps, enable_on_last_step_if_periodic=True):
                return

            results_tree = self._metric_collector.collect_results()
            results_flat = _flatten_pytree_for_metrics(results_tree)

            for name, value in results_flat.items():
                run.scalar(name, value)

    def state_dict(self) -> dict[str, Any]:
        return {
            "tracker": state_dict_main_process(self._dist_context, self._tracker),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        load_state_dict_main_process(self._dist_context, self._tracker, state_dict["tracker"])
