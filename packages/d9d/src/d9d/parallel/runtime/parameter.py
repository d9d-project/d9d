"""Sharded and unsharded parameters, including shared ownership and dtype conversion."""

import dataclasses
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor

from d9d.pipelining.api.groups import group_modules


@dataclasses.dataclass(slots=True, eq=False)
class ParameterPlan:
    """One parameter and the modules that hold it, before the runtime allocates anything for it.

    Attributes:
        owners: The modules that hold the parameter and its name in each. A tied parameter has several.
        sharded: The parameter that the optimizer and checkpoints see: a ``DTensor``, or a plain tensor outside a
            device mesh.
        trainable: Whether the job computes the gradient of the parameter.
        compute_dtype: The dtype that groups compute with.
        grad_dtype: The dtype of the gradient.
    """

    owners: list[tuple[nn.Module, str]]
    sharded: nn.Parameter
    trainable: bool
    compute_dtype: torch.dtype
    grad_dtype: torch.dtype


@dataclasses.dataclass(slots=True, eq=False)
class ParameterHandle:
    """One parameter while the runtime is bound: its plan and the unsharded parameter that groups compute with.

    Attributes:
        plan: The plan of the parameter.
        unsharded: The local parameter that groups compute with. Autograd writes its gradient.
        has_own_copy: Whether ``unsharded`` has its own memory in the compute dtype, which needs a refresh after the
            optimizer step.
        used: Whether a backward pass of this rank wrote the gradient in the current step.
    """

    plan: ParameterPlan
    unsharded: nn.Parameter
    has_own_copy: bool
    used: bool = False


def local_tensor(data: Tensor) -> Tensor:
    """Returns the local part of a ``DTensor``, or the tensor itself.

    Args:
        data: The data of a sharded parameter.

    Returns:
        The tensor that this rank holds.
    """
    return data.to_local() if isinstance(data, DTensor) else data


def grad_like(data: Tensor, local_grad: Tensor) -> Tensor:
    """Wraps a local gradient like the data of a parameter.

    Args:
        data: The data of the sharded parameter.
        local_grad: The local gradient of this rank.

    Returns:
        A ``DTensor`` with the shape, stride and placements of ``data`` that views ``local_grad``, or
        ``local_grad`` itself for a plain tensor.
    """
    if not isinstance(data, DTensor):
        return local_grad
    return DTensor.from_local(
        local_grad, shape=data.shape, stride=data.stride(), device_mesh=data.device_mesh, placements=data.placements
    )


def new_unsharded(plan: ParameterPlan, data: Tensor) -> nn.Parameter:
    """Creates the unsharded parameter of a plan over the given data.

    Args:
        plan: The plan of the parameter.
        data: The memory of the unsharded parameter.

    Returns:
        The unsharded parameter. Its gradient keeps the gradient dtype of the plan.
    """
    unsharded = nn.Parameter(data, requires_grad=plan.sharded.requires_grad)
    unsharded.grad_dtype = plan.grad_dtype
    return unsharded


def make_local_handle(plan: ParameterPlan) -> ParameterHandle:
    """Creates the handle of a parameter without a gather.

    Args:
        plan: The plan of the parameter.

    Returns:
        A handle whose unsharded parameter views the local data, or holds a copy of it in the compute dtype.
    """
    local = local_tensor(plan.sharded.data).detach()
    if local.dtype == plan.compute_dtype:
        # The unsharded parameter views the local data, so optimizer writes reach it directly.
        return ParameterHandle(plan=plan, unsharded=new_unsharded(plan, local), has_own_copy=False)
    return ParameterHandle(plan=plan, unsharded=new_unsharded(plan, local.to(plan.compute_dtype)), has_own_copy=True)


def install_tensor_hook_bridge(handle: ParameterHandle):
    """Makes the tensor hooks of the sharded parameter run on the gradient of the unsharded one.

    Autograd computes the gradient of the unsharded parameter only, so the hooks see the local gradient of each
    backward pass, before any reduction. The hooks are read at every call, so a hook that the user removes or adds
    later takes effect at once.

    Args:
        handle: The parameter.
    """
    if not handle.unsharded.requires_grad:
        return
    sharded = handle.plan.sharded

    def _run_hooks(grad: Tensor) -> Tensor:
        for hook in list((sharded._backward_hooks or {}).values()):  # noqa: SLF001 - PyTorch has no public reader of hooks
            result = hook(grad)
            if result is not None:
                grad = result
        return grad

    handle.unsharded.register_hook(_run_hooks)


def install_usage_hook(handle: ParameterHandle):
    """Marks the parameter as used when autograd writes the gradient of the unsharded one.

    Args:
        handle: The parameter.
    """
    if not handle.unsharded.requires_grad:
        return

    def _mark_used(_: Tensor):
        handle.used = True

    handle.unsharded.register_post_accumulate_grad_hook(_mark_used)


def run_post_accumulate_hooks(handle: ParameterHandle):
    """Runs the post-accumulate-grad hooks of the sharded parameter, once its gradient is complete.

    Args:
        handle: The parameter.
    """
    sharded = handle.plan.sharded
    # Mirrors FSDP2, which runs these hooks on the sharded parameter after the reduction. A hook can remove itself.
    hooks = list((sharded._post_accumulate_grad_hooks or {}).values())  # noqa: SLF001 - no public reader
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
            # The swap runs around every group, so it skips the checks of nn.Module.
            module._parameters[name] = parameter  # noqa: SLF001 - see above


def _make_plan(param: nn.Parameter, compute_dtype: torch.dtype | None, training: bool) -> ParameterPlan:
    local_dtype = local_tensor(param.data).dtype
    return ParameterPlan(
        owners=[],
        sharded=param,
        trainable=training and param.requires_grad,
        compute_dtype=compute_dtype or local_dtype,
        # Gradients keep the precision of the storage, whatever dtype the compute weights use.
        grad_dtype=param.grad_dtype or param.dtype,
    )


def plan_parameters(
    module: nn.Module, groups: Sequence[nn.Module], compute_dtype: torch.dtype | None, training: bool
) -> list[list[ParameterPlan]]:
    """Plans the parameters that each group uses.

    A tied parameter has one plan in every group that uses it, so autograd adds up its gradients.

    Args:
        module: The stage module that owns the parameters.
        groups: The groups of a stage.
        compute_dtype: The dtype of unsharded parameters, or ``None`` to keep the dtype of the storage.
        training: Whether the job computes gradients.

    Returns:
        The parameters of each group, by group index.
    """
    known: dict[int, ParameterPlan] = {}
    # A module can belong to several groups, but each of its slots holds the parameter once.
    slots: set[tuple[int, str]] = set()
    plans: list[list[ParameterPlan]] = []

    for group in groups:
        group_plans: dict[int, ParameterPlan] = {}
        for submodule in group_modules(group, set(groups)):
            for name, param in submodule.named_parameters(recurse=False, remove_duplicate=False):
                if id(param) not in known:
                    known[id(param)] = _make_plan(param, compute_dtype, training)
                plan = known[id(param)]

                if (id(submodule), name) not in slots:
                    slots.add((id(submodule), name))
                    plan.owners.append((submodule, name))
                group_plans[id(param)] = plan
        plans.append(list(group_plans.values()))

    # A slot of the stage module that no group module reaches must follow every swap too, e.g. when a group
    # module holds the parameter itself.
    for submodule in module.modules():
        for name, param in submodule.named_parameters(recurse=False, remove_duplicate=False):
            if (id(submodule), name) not in slots:
                slots.add((id(submodule), name))
                known[id(param)].owners.append((submodule, name))

    return plans


@torch.no_grad()
def refresh_own_copies(handles: Iterable[ParameterHandle]):
    """Copies the sharded weights into the unsharded parameters that have their own memory.

    Args:
        handles: The parameters, each once.
    """
    for handle in handles:
        if handle.has_own_copy:
            handle.unsharded.data.copy_(local_tensor(handle.plan.sharded.data))


def convert_buffers(modules: Iterable[nn.Module], compute_dtype: torch.dtype):
    """Converts the non-persistent floating-point buffers of modules to the compute dtype.

    Persistent buffers are training state, such as the expert bias of a router, so they keep their dtype. A buffer
    that several modules hold stays shared.

    Args:
        modules: The modules.
        compute_dtype: The dtype of the computation.
    """
    converted: dict[int, Tensor] = {}
    for module in modules:
        for submodule in module.modules():
            non_persistent = submodule._non_persistent_buffers_set  # noqa: SLF001 - PyTorch has no public reader
            for name, buffer in submodule.named_buffers(recurse=False):
                if buffer.is_floating_point() and name in non_persistent:
                    if id(buffer) not in converted:
                        converted[id(buffer)] = buffer.to(compute_dtype)
                    submodule.register_buffer(name, converted[id(buffer)], persistent=False)
