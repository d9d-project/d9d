from typing import Any

from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle


def _build_to_local_patched_class(
    base_class: type, grad_placement: tuple[Placement, ...], param_names: tuple[str, ...]
) -> type:
    param_name_to_property = {
        param_name: property(lambda self, pn=param_name: self._parameters[pn].to_local(grad_placements=grad_placement))
        for param_name in param_names
    }
    return type(
        f"Replicate{base_class.__name__}",
        (base_class,),
        param_name_to_property,
    )


class _LocalParameterAccess:
    """Swaps the submodule classes for the forward pass, so that parameter attributes return local tensors."""

    def __init__(self, param_names: dict[str, tuple[str, ...]], grad_placement: tuple[Placement, ...]):
        self._param_names = param_names
        self._grad_placement = grad_placement
        self._patched_classes: dict[tuple[type, tuple[str, ...]], type] = {}
        # One entry per forward pass in progress: the submodules swapped so far and their classes before the swap.
        self._swaps: list[list[tuple[nn.Module, type]]] = []

    def _patched_class(self, base_class: type, param_names: tuple[str, ...]) -> type:
        key = (base_class, param_names)
        if key not in self._patched_classes:
            self._patched_classes[key] = _build_to_local_patched_class(base_class, self._grad_placement, param_names)
        return self._patched_classes[key]

    def patch(self, module: nn.Module, args: Any):
        swaps: list[tuple[nn.Module, type]] = []
        # Register the entry before any swap, so restore() undoes exactly the swaps done if this hook fails.
        self._swaps.append(swaps)
        for submod_name, submod in module.named_modules():
            # Patch the current class, not the class at parallelization time: styles applied later, such as
            # fully_shard(), swap the class too, and restore() must keep their class.
            base_class = submod.__class__
            submod.__class__ = self._patched_class(base_class, self._param_names[submod_name])
            swaps.append((submod, base_class))

    def restore(self, module: nn.Module, args: Any, output: Any):
        # PyTorch calls this hook also when an earlier forward pre-hook raised before patch() ran.
        if not self._swaps:
            return
        for submod, base_class in self._swaps.pop():
            submod.__class__ = base_class


class ToLocalParallel(ParallelStyle):
    """Parallel style that stores parameters as ``DTensor`` but computes with local tensors.

    Parameters become ``DTensor`` objects with the given placements. During the forward pass, the
    module sees them as local ``torch.Tensor`` objects. Use it for strategies such as Replicate, where
    the module code is not ``DTensor``-aware but gradient sync and checkpointing need ``DTensor``.
    """

    def __init__(self, param_placement: tuple[Placement, ...], grad_placement: tuple[Placement, ...]):
        """Constructs the ``ToLocalParallel`` object.

        Args:
            param_placement: The placements of the parameters, one per mesh dimension.
            grad_placement: The placements of the gradients, one per mesh dimension. They are passed to
                ``DTensor.to_local`` as ``grad_placements``.
        """
        self._grad_placement = grad_placement
        self._param_placement = param_placement

    def _distribute_params(self, name: str, module: nn.Module, device_mesh: DeviceMesh):
        for param_name, param in module.named_parameters(recurse=False):
            new_param = nn.Parameter(
                distribute_tensor(param.data, device_mesh, self._param_placement), requires_grad=param.requires_grad
            )

            module.register_parameter(param_name, new_param)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh):
        param_names = {}

        for submod_name, submod in module.named_modules():
            param_names[submod_name] = tuple(name for name, p in submod.named_parameters(recurse=False))

            distribute_module(submod, device_mesh, self._distribute_params)

        # Swap each submodule to a subclass whose parameter attributes return local tensors for the
        # forward pass. Restore the classes afterwards, so parameter access returns DTensors again.
        access = _LocalParameterAccess(param_names, self._grad_placement)
        module.register_forward_pre_hook(access.patch)
        # A forward pass that raises must not leave the patched classes behind.
        module.register_forward_hook(access.restore, always_call=True)
