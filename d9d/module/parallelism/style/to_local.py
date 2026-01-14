from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, distribute_tensor, distribute_module
from torch.distributed.tensor.parallel import ParallelStyle


def _build_to_local_patched_class(
        module: nn.Module,
        grad_placement: tuple[Placement, ...],
        param_names: list[str]
) -> type:
    param_name_to_property = {
        param_name: property(
            lambda self, pn=param_name: self._parameters[pn].to_local(grad_placements=grad_placement)
        )
        for param_name in param_names
    }
    return type(
        f"Replicate{module.__class__.__name__}",
        (module.__class__,),
        param_name_to_property,
    )


class _ModulePatch:
    def __init__(self, class_mapper: dict[str, type]):
        self._class_mapper = class_mapper

    def __call__(self, mod: nn.Module, *args, **kwargs):
        for submod_name, submod in mod.named_modules():
            submod.__class__ = self._class_mapper[submod_name]


class ToLocalParallel(ParallelStyle):
    """
    Parallel style that distributes parameters and gradients but executes with local tensors.

    This style wraps standard tensor distribution (via ``DTensor``) but injects
    runtime hooks to temporarily unwrap ``DTensor`` parameters into local ``torch.Tensor``
    during the forward pass.

    This is useful for parallel strategies (like Replicate)
    where the underlying calculation logic is not DTensor-aware, but the parameters must remain
    distributed for gradient synchronization and for distributed checkpointing.
    """

    def __init__(self, param_placement: tuple[Placement, ...], grad_placement: tuple[Placement, ...]):
        """
        Constructs ToLocalParallel object.

        Args:
            param_placement: Tuple of placements defining how parameters are distributed.
            grad_placement: Tuple of placements defining how gradients are synchronized.
        """

        self._grad_placement = grad_placement
        self._param_placement = param_placement

    def _distribute_params(self, name: str, module: nn.Module, device_mesh: DeviceMesh):
        for param_name, param in module.named_parameters(recurse=False):
            new_param = nn.Parameter(
                distribute_tensor(param.data, device_mesh, self._param_placement),
                requires_grad=param.requires_grad
            )

            module.register_parameter(param_name, new_param)

    def _apply(self, master_module: nn.Module, device_mesh: DeviceMesh):
        patched_classes = {}
        original_classes = {}

        for submod_name, submod in master_module.named_modules():
            param_names = [name for name, p in submod.named_parameters(recurse=False)]
            patched_classes[submod_name] = _build_to_local_patched_class(submod, self._grad_placement, param_names)
            original_classes[submod_name] = submod.__class__

            distribute_module(
                submod,
                device_mesh,
                self._distribute_params
            )


        master_module.register_forward_pre_hook(_ModulePatch(patched_classes))
        master_module.register_forward_hook(_ModulePatch(original_classes))

