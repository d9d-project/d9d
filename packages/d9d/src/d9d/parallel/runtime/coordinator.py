from collections.abc import Collection, Sequence

from torch import nn

from .bound import BoundParameters
from .bucket import SideStreams
from .module import BoundModule, bind_module
from .plan import module_path, plan_modules


class ParameterRuntime:
    """The registered modules and the configuration of data parallelism that its caller drives region by region.

    A region is a part of the forward pass that needs the parameters of one group.
    """

    def __init__(self, gather_dims: Collection[str], bucket_size_mib: int, training: bool):
        """Constructs the ``ParameterRuntime`` object.

        Args:
            gather_dims: The mesh dimensions along which data parallelism shards parameters. A ``Shard`` along any other
                dimension stays for its layer.
            bucket_size_mib: The size limit of buckets without a gather, in MiB.
            training: Whether the job computes gradients.

        Raises:
            ValueError: If ``bucket_size_mib`` is not positive.
        """
        if bucket_size_mib <= 0:
            raise ValueError(f"The bucket size bucket_size_mib ({bucket_size_mib}) must be positive.")
        self._gather_dims = frozenset(gather_dims)
        self._bucket_size_mib = bucket_size_mib
        self._training = training
        self._group_roots: dict[nn.Module, tuple[nn.Module, ...]] = {}
        self._bound: BoundParameters | None = None

    def _is_bound(self) -> bool:
        return self._bound is not None and not self._bound.closed

    def register(self, module: nn.Module, group_roots: Sequence[nn.Module]):
        """Registers a module and the roots of its parameter groups.

        Args:
            module: The module, the root of its own group, which holds the parameters outside ``group_roots``.
            group_roots: The roots of the other groups, inside ``module``, or empty.

        Raises:
            ValueError: If the runtime is bound, if ``module`` is already registered, or if a root is repeated, is
                ``module`` or lies outside it.
        """
        if self._is_bound():
            raise ValueError(
                f"The runtime is bound, so it cannot register the module ({type(module).__name__}). Close the "
                "BoundParameters first."
            )
        if module in self._group_roots:
            raise ValueError(f"The module ({type(module).__name__}) is already registered. Register each module once.")
        submodules = set(module.modules()) - {module}
        seen: set[nn.Module] = set()
        for root in group_roots:
            if root in seen:
                raise ValueError(
                    f"The group root ({module_path(module, root)}) of the module ({type(module).__name__}) is "
                    "repeated. "
                    "Pass each root once."
                )
            if root not in submodules:
                raise ValueError(
                    f"The group root ({type(root).__name__}) is not a submodule of the module "
                    f"({type(module).__name__}). Pass the roots inside the module."
                )
            seen.add(root)
        self._group_roots[module] = tuple(group_roots)

    def bind(self) -> BoundParameters:
        """Builds the unsharded parameters and the buffers of the registered modules.

        The steps must run on the current stream.

        Returns:
            The bound parameters.

        Raises:
            ValueError: If the runtime is already bound, if a parameter is shared by several modules, if a trainable
                parameter has no ``grad_dtype``, if the placements of a parameter are not supported, or if a shard does
                not match ``torch.chunk``.
        """
        if self._is_bound():
            raise ValueError("The runtime is already bound. Close the BoundParameters first.")

        plans = plan_modules(self._group_roots, self._training, self._bucket_size_mib, self._gather_dims)
        streams = SideStreams()
        modules: dict[nn.Module, BoundModule] = {}
        try:
            for module, plan in plans.items():
                modules[module] = bind_module(module, plan, streams)
        except Exception:
            # A failed module releases what it built, and nothing queued work on the side streams yet.
            for bound in modules.values():
                bound.close()
            raise
        self._bound = BoundParameters(streams, modules)
        return self._bound
