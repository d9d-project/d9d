import dataclasses
from collections.abc import Sequence

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from ..plan import ParameterPlan, local_tensor


@dataclasses.dataclass(slots=True, eq=False)
class ParameterHandle:
    """One parameter while it is bound: its plan and the unsharded parameter that regions compute with.

    Attributes:
        plan: The plan of the parameter.
        unsharded: The local parameter that regions compute with. Autograd writes its gradient.
        hook: The hook that runs the tensor hooks of the sharded parameter, or ``None`` without one.
    """

    plan: ParameterPlan
    unsharded: nn.Parameter
    hook: RemovableHandle | None


def new_unsharded(plan: ParameterPlan, data: Tensor) -> nn.Parameter:
    """Creates the unsharded parameter of a plan over a tensor.

    Args:
        plan: The plan of the parameter.
        data: The tensor that the unsharded parameter wraps.

    Returns:
        The unsharded parameter. It requires a gradient only if the plan is trainable, and its gradient has the
        ``grad_dtype`` of the sharded parameter.
    """
    # Without training, a backward pass for input gradients must not leave gradients on the unsharded parameters.
    unsharded = nn.Parameter(data, requires_grad=plan.trainable)
    unsharded.grad_dtype = plan.sharded.grad_dtype
    return unsharded


def make_local_handle(plan: ParameterPlan) -> ParameterHandle:
    """Creates the handle of a parameter without a gather.

    Args:
        plan: The plan of the parameter.

    Returns:
        The handle. Its unsharded parameter views the local data.
    """
    local = local_tensor(plan.sharded.data).detach()
    return ParameterHandle(plan=plan, unsharded=new_unsharded(plan, local), hook=None)


def install_tensor_hook_bridge(handle: ParameterHandle):
    """Runs the tensor hooks of the sharded parameter on each gradient of the unsharded one.

    The hooks see the local gradient of each backward pass, before any reduction. Hooks that are added or removed later
    take effect at once. Without gradients, it installs nothing.

    Args:
        handle: The parameter. Its ``hook`` gets the installed hook.
    """
    if not handle.unsharded.requires_grad:
        return
    sharded = handle.plan.sharded

    # The hooks are read at every call, so the bridge follows later changes to them.
    def _run_hooks(grad: Tensor) -> Tensor:
        # ruff: ignore[private-member-access] - PyTorch has no public reader of hooks
        for hook in list((sharded._backward_hooks or {}).values()):
            result = hook(grad)
            if result is not None:
                grad = result
        return grad

    handle.hook = handle.unsharded.register_hook(_run_hooks)


def run_post_accumulate_hooks(handle: ParameterHandle):
    """Runs the post-accumulate-grad hooks of the sharded parameter.

    Args:
        handle: The parameter.
    """
    sharded = handle.plan.sharded
    # A copy of the hooks, because a hook can remove itself.
    # ruff: ignore[private-member-access] - no public reader
    hooks = list((sharded._post_accumulate_grad_hooks or {}).values())
    for hook in hooks:
        hook(sharded)


def bind_parameters(handles: Sequence[ParameterHandle], unsharded: bool):
    """Puts the unsharded or the sharded parameters into every module that holds them.

    Args:
        handles: The parameters.
        unsharded: Whether to put the unsharded parameters. Otherwise, it puts the sharded ones back.
    """
    for handle in handles:
        parameter = handle.unsharded if unsharded else handle.plan.sharded
        for module, name in handle.plan.owners:
            # ruff: ignore[private-member-access] - the swap runs around every region, so it skips the nn.Module checks
            module._parameters[name] = parameter


def release_parameter(handle: ParameterHandle):
    """Removes the hook of a parameter and frees the data and the gradient of its unsharded parameter.

    Args:
        handle: The parameter.
    """
    if handle.hook is not None:
        handle.hook.remove()
    # A graph that the caller still holds, e.g. through a logged loss, keeps the unsharded parameter alive. So it
    # must keep neither the gradient buffer nor the local data, which an offload replaces.
    handle.unsharded.grad = None
    handle.unsharded.data = handle.unsharded.data.new_empty(0)
