from contextlib import contextmanager

from d9d.core.dist_context import REGULAR_DOMAIN, DistributedContext
from d9d.internals.grad_norm import ParametersForNorm, clip_grad_norm_distributed_, group_parameters_for_norm
from d9d.loop.config import GradientClippingConfig
from d9d.tracker import BaseTrackerRun

from .model_stage_factory import TrackedModules
from .stepper import Stepper


class GradientClipper:
    """
    Manages gradient clipping and logging of gradient norms in a distributed execution environment.
    """

    def __init__(
            self,
            dist_context: DistributedContext,
            tracked_modules: TrackedModules,
            config: GradientClippingConfig,
            stepper: Stepper
    ):
        """
        Constructs the gradient clipper.

        Args:
            dist_context: The distributed context.
            tracked_modules: Container of model modules whose parameters need clipping.
            config: Configuration defining max norm and logging frequency.
            stepper: Stepper instance used to track the current training step.
        """

        self._dist_context = dist_context
        self._tracked_modules = tracked_modules
        self._config = config
        self._stepper = stepper

        self._parameter_groups: ParametersForNorm | None = None

    def _all_parameters(self):
        for model in self._tracked_modules.modules:
            yield from model.parameters()

    @contextmanager
    def install(self):
        """
        Context manager that prepares and groups parameters for efficient norm calculation.

        It calculates necessary metadata (such as segregating shared parameters) to ensure
        correct global norm calculation across the pipeline parallel mesh.
        """

        self._parameter_groups = group_parameters_for_norm(self._all_parameters())
        yield
        self._parameter_groups = None

    def clip_and_log(self, run: BaseTrackerRun):
        """
        Clips gradients to the configured maximum norm and logs the total L2 norm.

        This method performs an in-place modification of parameter gradients if a
        maximum norm is configured. It calculates the global gradient norm across
        distributed ranks.

        Args:
            run: The tracker run instance used for logging the norm scalar.

        Raises:
            ValueError: If called outside the ``install`` context manager scope.
        """

        should_log = self._stepper.should_do_action(self._config.log_total_steps)

        if not self._config.max_norm and not should_log:
            return

        if self._parameter_groups is None:
            raise ValueError("Parameter groups are not configured")

        grad_norm = clip_grad_norm_distributed_(
            parameter_groups=self._parameter_groups,
            max_norm=self._config.max_norm,
            norm_type=2.0,
            pp_mesh=self._dist_context.mesh_for(REGULAR_DOMAIN)["pp"],
        )

        if should_log:
            run.scalar(name="l2_grad_norm_total", value=grad_norm.item())
