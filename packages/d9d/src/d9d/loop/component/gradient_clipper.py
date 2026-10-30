from contextlib import contextmanager

import torch

from d9d.core.dist_context import REGULAR_DOMAIN, DistributedContext
from d9d.internals.grad_norm import ParametersForNorm, clip_grad_norm_distributed_, group_parameters_for_norm
from d9d.loop.config import GradientClippingConfig

from .model_stage_factory import TrackedModules


class GradientClipper:
    """Gradient clipper that computes the global gradient norm in a distributed execution environment."""

    def __init__(
        self,
        dist_context: DistributedContext,
        tracked_modules: TrackedModules,
        config: GradientClippingConfig,
    ):
        """Constructs the ``GradientClipper`` object.

        Args:
            dist_context: The distributed context.
            tracked_modules: The model stages whose gradients are clipped.
            config: The configuration that sets the maximum norm.
        """
        self._dist_context = dist_context
        self._tracked_modules = tracked_modules
        self._config = config

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

    def clip(self) -> torch.Tensor:
        """Computes the global L2 norm of the gradients and clips them to the configured maximum norm.

        Gradients are modified in place if a maximum norm is configured. The norm is global across all ranks.

        Returns:
            The total L2 norm of the gradients before clipping, as a scalar tensor on the device.

        Raises:
            ValueError: If called outside the ``install`` context manager scope.
        """
        if self._parameter_groups is None:
            raise ValueError("clip() must be called inside the install() context.")

        if self._dist_context.mesh_params.is_distributed:
            pp_mesh = self._dist_context.mesh_for(REGULAR_DOMAIN)["pp"]
        else:
            pp_mesh = None

        return clip_grad_norm_distributed_(
            parameter_groups=self._parameter_groups,
            max_norm=self._config.max_norm,
            norm_type=2.0,
            pp_mesh=pp_mesh,
        )
