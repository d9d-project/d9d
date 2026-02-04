from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
import torch.utils._pytree as pytree  # noqa: PLC2701
from torch.distributed.checkpoint.stateful import Stateful

from d9d.core.dist_context import DistributedContext
from d9d.core.types import PyTree, ScalarTree
from d9d.internals.state import load_state_dict_main_process, state_dict_main_process
from d9d.loop.config import JobLoggerConfig
from d9d.metric.impl import ComposeMetric
from d9d.tracker import BaseTracker, BaseTrackerRun, RunConfig, tracker_from_config
from d9d.tracker.provider.null import NullTrackerConfig

from .stepper import Stepper


def _flatten_pytree_for_metrics(tree: PyTree[float]) -> dict[str, float]:
    flat_dict = {}

    for path_tuple, value in pytree.tree_leaves_with_path(tree):
        path_segments = []

        for key in path_tuple:
            match key:
                case pytree.MappingKey(k):
                    path_segments.append(str(k))
                case pytree.SequenceKey(idx):
                    path_segments.append(str(idx))
                case pytree.GetAttrKey(name):
                    path_segments.append(name)
                case _:
                    path_segments.append(str(key))

        flat_key = "/".join(path_segments)
        flat_dict[flat_key] = value

    return flat_dict


class JobLogger(Stateful):
    """
    Handles the logging of training metrics and loss values.

    This class coordinates with the distributed context and metric calculators
    to log instantaneous loss values and periodic aggregated metrics to the
    configured experiment tracker.
    """

    def __init__(
            self,
            dist_context: DistributedContext,
            config: JobLoggerConfig,
            metrics: ComposeMetric,
            stepper: Stepper,
            run_config: RunConfig,
            additional_hparams: ScalarTree
    ):
        """
        Constructs JobLogger object.

        Args:
            dist_context: The distributed context.
            config: Configuration settings.
            metrics: The composite metric collection to be computed and logged.
            stepper: Object tracking the current global step.
            run_config: Run configuration.
        """

        self._dist_context = dist_context
        self._config = config
        self._metrics = metrics
        self._stepper = stepper
        self._run_config = run_config.model_copy(deep=True, update={"hparams": {
            "run": run_config.hparams,
            "params": additional_hparams
        }})

        self._tracker = self._build_tracker()

    def _build_tracker(self) -> BaseTracker:
        if self._dist_context.is_main_process:
            return tracker_from_config(self._config.tracker)
        else:
            return tracker_from_config(NullTrackerConfig())

    @contextmanager
    def new_run(self) -> Generator[BaseTrackerRun, None, None]:
        with self._tracker.open(self._run_config) as run:
            yield run

    def trigger_sync(self):
        """
        Conditionally initiates the synchronization of distributed metrics.

        Checks if the current step is scheduled for metric logging. If so, it
        triggers the asynchronous communication required to aggregate metric values
        across ranks. This allows communication to overlap with other operations
        before `log` is called.
        """

        if not self._stepper.should_do_action(self._config.period_steps, enable_on_last_step_if_periodic=True):
            return

        self._metrics.trigger_sync(self._dist_context)

    def log(self, run: BaseTrackerRun, loss_value: torch.Tensor):
        """
        Logs the current loss and conditionally processes aggregated metrics.

        This method always logs the provided loss value. Periodically (determined
        by the stepper and configuration), it waits for the synchronization of
        metrics to complete (initiated by `trigger_sync`), computes their values,
        flattens the result structure, logs them to the tracker, and resets the
        metrics for the next window.

        Args:
            run: The active tracker run interface for sending data.
            loss_value: Tensor containing the scalar loss for the current step.
        """

        run.scalar("loss", loss_value.item())

        if not self._stepper.should_do_action(self._config.period_steps, enable_on_last_step_if_periodic=True):
            return

        self._metrics.wait_sync(self._dist_context)

        results_tree = self._metrics.compute()
        results_tree = pytree.tree_map(lambda x: x.item(), results_tree)
        results_flat = _flatten_pytree_for_metrics(results_tree)

        for name, value in results_flat.items():
            run.scalar(name, value)

        self._metrics.reset()

    def state_dict(self) -> dict[str, Any]:
        return {
            "tracker": state_dict_main_process(self._dist_context, self._tracker),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        load_state_dict_main_process(self._dist_context, self._tracker, state_dict["tracker"])
