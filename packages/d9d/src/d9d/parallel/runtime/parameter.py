"""Sharded and unsharded parameters and the modules that hold them."""

import dataclasses
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor
from torch.utils.hooks import RemovableHandle


@dataclasses.dataclass(slots=True, eq=False)
class ParameterPlan:
    """One parameter and the modules that hold it, before the runtime allocates anything for it.

    Attributes:
        owners: The modules that hold the parameter and its name in each. A tied parameter has several.
        sharded: The parameter that the optimizer and checkpoints see: a ``DTensor``, or a plain tensor outside a
            device mesh.
        trainable: Whether the job computes the gradient of the parameter.
        grad_dtype: The dtype of the gradient.
    """

    owners: list[tuple[nn.Module, str]]
    sharded: nn.Parameter
    trainable: bool
    grad_dtype: torch.dtype

    @property
    def name(self) -> str:
        """The name of the parameter in its first owner, e.g. ``Linear.weight``."""
        module, name = self.owners[0]
        return f"{type(module).__name__}.{name}"


@dataclasses.dataclass(slots=True, eq=False)
class ParameterHandle:
    """One parameter while the runtime is bound: its plan and the unsharded parameter that groups compute with.

    Attributes:
        plan: The plan of the parameter.
        unsharded: The local parameter that groups compute with. Autograd writes its gradient.
        hook: The hook that runs the tensor hooks of the sharded parameter, or ``None`` without gradients.
    """

    plan: ParameterPlan
    unsharded: nn.Parameter
    hook: RemovableHandle | None = None


def local_tensor(data: Tensor) -> Tensor:
    """Returns the local part of a ``DTensor``, or the tensor itself.

    Args:
        data: The data of a sharded parameter.

    Returns:
        The tensor that this rank holds.
    """
    return data.to_local() if isinstance(data, DTensor) else data


def new_unsharded(plan: ParameterPlan, data: Tensor) -> nn.Parameter:
    """Creates the unsharded parameter of a plan over the given data.

    Args:
        plan: The plan of the parameter.
        data: The memory of the unsharded parameter.

    Returns:
        The unsharded parameter. Its gradient keeps the gradient dtype of the plan.
    """
    # Without training, a backward pass for input gradients must not leave gradients on the unsharded parameters.
    unsharded = nn.Parameter(data, requires_grad=plan.trainable)
    unsharded.grad_dtype = plan.grad_dtype
    return unsharded


def make_local_handle(plan: ParameterPlan) -> ParameterHandle:
    """Creates the handle of a parameter without a gather.

    Args:
        plan: The plan of the parameter.

    Returns:
        A handle whose unsharded parameter views the local data, so optimizer writes reach it directly.
    """
    local = local_tensor(plan.sharded.data).detach()
    return ParameterHandle(plan=plan, unsharded=new_unsharded(plan, local))


def install_tensor_hook_bridge(handle: ParameterHandle):
    """Makes the tensor hooks of the sharded parameter run on the gradient of the unsharded one.

    Autograd computes the gradient of the unsharded parameter only, so the hooks see the local gradient of each
    backward pass, before any reduction. The hooks are read at every call, so a hook that the user removes or adds
    later takes effect at once.

    Args:
        handle: The parameter. Its ``hook`` gets the handle of the installed hook.
    """
    if not handle.unsharded.requires_grad:
        return
    sharded = handle.plan.sharded

    def _run_hooks(grad: Tensor) -> Tensor:
        # ruff: ignore[private-member-access] - PyTorch has no public reader of hooks
        for hook in list((sharded._backward_hooks or {}).values()):
            result = hook(grad)
            if result is not None:
                grad = result
        return grad

    handle.hook = handle.unsharded.register_hook(_run_hooks)


def run_post_accumulate_hooks(handle: ParameterHandle):
    """Runs the post-accumulate-grad hooks of the sharded parameter, once its gradient is complete.

    Args:
        handle: The parameter.
    """
    sharded = handle.plan.sharded
    # The gradient of the sharded parameter is complete only after the final reduction. A hook can remove itself.
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


def _make_plan(param: nn.Parameter, training: bool) -> ParameterPlan:
    return ParameterPlan(
        owners=[],
        sharded=param,
        trainable=training and param.requires_grad,
        grad_dtype=param.grad_dtype or param.dtype,
    )


def _group_modules(root: nn.Module, roots: set[nn.Module]) -> list[nn.Module]:
    result: dict[nn.Module, None] = {}
    # A loop instead of a nested recursive function, which would hold the modules in a reference cycle.
    pending = [root]
    while pending:
        current = pending.pop()
        if current in result:
            continue
        result[current] = None
        pending.extend(child for child in reversed(list(current.children())) if child not in roots)
    return list(result)


def plan_parameters(groups: Sequence[nn.Module], training: bool) -> list[list[ParameterPlan]]:
    """Plans the parameters that each group uses.

    A group holds the parameters of its root and of the root's descendants, except those of other roots.

    A tied parameter has one plan, listed in every group that uses it, so autograd adds up its gradients.

    Args:
        groups: The roots of the groups of a stage. The first one is the stage root.
        training: Whether the job computes gradients.

    Returns:
        The parameters of each group, by group index.
    """
    known: dict[int, ParameterPlan] = {}
    # A module can belong to several groups, but each of its slots holds the parameter once.
    slots: set[tuple[nn.Module, str]] = set()
    plans: list[list[ParameterPlan]] = []

    for group in groups:
        group_plans: dict[int, ParameterPlan] = {}
        for submodule in _group_modules(group, set(groups)):
            for name, param in submodule.named_parameters(recurse=False, remove_duplicate=False):
                if id(param) not in known:
                    known[id(param)] = _make_plan(param, training)
                plan = known[id(param)]

                if (submodule, name) not in slots:
                    slots.add((submodule, name))
                    plan.owners.append((submodule, name))
                group_plans[id(param)] = plan
        plans.append(list(group_plans.values()))

    return plans
