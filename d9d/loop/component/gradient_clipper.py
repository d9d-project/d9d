from contextlib import contextmanager

from d9d.core.dist_context import REGULAR_DOMAIN, DistributedContext
from d9d.internals.grad_norm import ParametersForNorm, clip_grad_norm_distributed_, group_parameters_for_norm
from d9d.loop.config import GradientClippingConfig
from d9d.tracker import BaseTrackerRun

from .job_schedule import JobSchedule
from .model_stage_factory import TrackedModules


class GradientClipper:
    """Manages gradient clipping and logging of gradient norms in a distributed execution environment."""

    def __init__(
        self,
        dist_context: DistributedContext,
        tracked_modules: TrackedModules,
        config: GradientClippingConfig,
        schedule: JobSchedule,
    ):
        """Constructs the ``GradientClipper`` object.

        Args:
            dist_context: The distributed context.
            tracked_modules: The model stages whose gradients are clipped.
            config: The configuration that sets the maximum norm and the logging period.
            schedule: The job schedule that tracks the current step.
        """
        self._dist_context = dist_context
        self._tracked_modules = tracked_modules
        self._config = config
        self._schedule = schedule

        self._parameter_groups: ParametersForNorm | None = None

    def _all_parameters(self):
        for model in self._tracked_modules.modules:
            yield from model.parameters()

    @contextmanager
    def install(self):
        """Groups the parameters for global gradient norm computation while the context is open."""
        self._parameter_groups = group_parameters_for_norm(self._all_parameters())
        yield
        self._parameter_groups = None

    def clip_and_log(self, run: BaseTrackerRun):
        """Clips gradients to the configured maximum norm and logs the total L2 norm.

        Gradients are modified in place if a maximum norm is configured. The norm is global across all ranks.

        Args:
            run: The tracker run instance used for logging the norm scalar.

        Raises:
            ValueError: If called outside the ``install`` context manager scope.
        """
        should_log = self._schedule.should_do_action(self._config.log_total_steps)

        if not self._config.max_norm and not should_log:
            return

        if self._parameter_groups is None:
            raise ValueError("clip_and_log() must be called inside the install() context.")

        if self._dist_context.mesh_params.is_distributed:
            pp_mesh = self._dist_context.mesh_for(REGULAR_DOMAIN)["pp"]
        else:
            pp_mesh = None

        grad_norm = clip_grad_norm_distributed_(
            parameter_groups=self._parameter_groups,
            max_norm=self._config.max_norm,
            norm_type=2.0,
            pp_mesh=pp_mesh,
        )

        if should_log:
            run.scalar(name="l2_grad_norm_total", value=grad_norm.item())
