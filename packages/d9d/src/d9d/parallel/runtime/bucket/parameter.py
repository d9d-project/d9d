"""Unsharded parameters while the runtime is bound, their hooks and the modules that hold them."""

import dataclasses
from collections.abc import Sequence

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from ..plan import ParameterPlan, local_tensor


@dataclasses.dataclass(slots=True, eq=False)
class ParameterHandle:
    """One parameter of a bound runtime: its plan and the unsharded parameter that regions compute with.

    Attributes:
        plan: The plan of the parameter.
        unsharded: The local parameter that regions compute with. Autograd writes its gradient.
        hook: The hook that runs the tensor hooks of the sharded parameter, or ``None`` before
            ``install_tensor_hook_bridge()`` and without gradients.
    """

    plan: ParameterPlan
    unsharded: nn.Parameter
    hook: RemovableHandle | None = None


def new_unsharded(plan: ParameterPlan, data: Tensor) -> nn.Parameter:
    """Creates the unsharded parameter of a plan.

    Args:
        plan: The plan of the parameter.
        data: The tensor that the unsharded parameter wraps.

    Returns:
        The unsharded parameter. It requires a gradient only if the plan is trainable, in the ``grad_dtype`` of the
        sharded parameter.
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
        A handle whose unsharded parameter views the local data, so it sees optimizer writes at once.
    """
    local = local_tensor(plan.sharded.data).detach()
    return ParameterHandle(plan=plan, unsharded=new_unsharded(plan, local))


def install_tensor_hook_bridge(handle: ParameterHandle):
    """Makes the tensor hooks of the sharded parameter run on the gradient of the unsharded one.

    The hooks see the local gradient of each backward pass, before any reduction. A hook that is added or removed
    later takes effect at once. Without gradients, it does nothing.

    Args:
        handle: The parameter. Its ``hook`` gets the handle of the installed hook.
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

    Call it after the final reduction.

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
    """Puts unsharded or sharded parameters into every module that holds them.

    Args:
        handles: The parameters.
        unsharded: Whether to put the unsharded parameters, otherwise the sharded ones.
    """
    for handle in handles:
        parameter = handle.unsharded if unsharded else handle.plan.sharded
        for module, name in handle.plan.owners:
            # ruff: ignore[private-member-access] - the swap runs around every region, so it skips the nn.Module checks
            module._parameters[name] = parameter


def release_parameter(handle: ParameterHandle):
    """Removes the hook bridge and leaves the unsharded parameter without data and gradient.

    Args:
        handle: The parameter.
    """
    if handle.hook is not None:
        handle.hook.remove()
    # A graph that the program still holds, e.g. through a logged loss, keeps the unsharded parameter alive. So it
    # must keep neither the gradient buffer nor the local data, which an offload replaces.
    handle.unsharded.grad = None
    handle.unsharded.data = handle.unsharded.data.new_empty(0)
