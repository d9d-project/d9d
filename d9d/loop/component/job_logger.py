from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful

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
    """Handles the logging of training metrics and loss values.

    This class coordinates with the distributed context and metric calculators
    to log instantaneous loss values and periodic aggregated metrics to the
    configured experiment tracker.
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
        """Constructs JobLogger object.

        Args:
            dist_context: The distributed context.
            config: Configuration settings.
            metrics: The composite metric collection to be computed and logged.
            schedule: Object tracking the current global step.
            run_config: Run configuration.
            additional_hparams: Supplemental hyperparameters to log for this run.
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
        """Creates a context manager for a new experiment run.

        Yields:
            The active tracker run interface.
        """
        with self._tracker.open(self._run_config) as run:
            yield run

    @contextmanager
    def install(self):
        """Prepares the metric collector resources (e.g., CUDA streams).

        This context manager ensures async metrics are bound to the device before
        usage and unbound afterwards.
        """
        self._metric_collector.bind()
        yield
        self._metric_collector.unbind()

    def trigger_sync(self):
        """Conditionally initiates the synchronization of distributed metrics.

        Checks if the current step is scheduled for metric logging. If so, it
        triggers the asynchronous communication required to aggregate metric values
        across ranks. This allows communication to overlap with other operations
        before `log` is called.
        """
        if not self._schedule.should_do_action(self._config.period_steps, enable_on_last_step_if_periodic=True):
            return

        self._metric_collector.schedule_collection(self._dist_context)

    def log(self, run: BaseTrackerRun, loss_value: torch.Tensor):
        """Logs the current loss and conditional metric results.

        This method always logs the provided loss value. Periodically (determined
        by the schedule configuration), it retrieves the asynchronous results from
        the metric collector (initiated by `trigger_sync`), flattens the result
        structure, and logs them to the tracker.

        Args:
            run: The active tracker run interface for sending data.
            loss_value: Tensor containing the scalar loss for the current step.
        """
        run.scalar("loss", loss_value.item())

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
