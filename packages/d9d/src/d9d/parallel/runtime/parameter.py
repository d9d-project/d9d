"""Sharded and unsharded parameters, including shared ownership and dtype conversion."""

import dataclasses
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor

from d9d.pipelining.api import StageSegment


@dataclasses.dataclass(slots=True, eq=False)
class ParameterHandle:
    """One parameter and the modules that hold it.

    Attributes:
        owners: The modules that hold the parameter and its name in each. A tied parameter has several.
        sharded: The parameter that the optimizer and checkpoints see: a ``DTensor``, or a plain tensor outside a
            device mesh.
        unsharded: The local parameter that segments compute with. Autograd writes its gradient.
        trainable: Whether the job computes the gradient of the parameter.
        has_own_copy: Whether ``unsharded`` has its own memory with another dtype, which needs a refresh after the
            optimizer step.
        shard_sizes: The number of elements of the shard of each rank, for a gathered parameter.
    """

    owners: list[tuple[nn.Module, str]]
    sharded: nn.Parameter
    unsharded: nn.Parameter
    trainable: bool
    has_own_copy: bool
    shard_sizes: list[int] = dataclasses.field(default_factory=list)


def local_tensor(data: Tensor) -> Tensor:
    """Returns the local part of a ``DTensor``, or the tensor itself.

    Args:
        data: The data of a sharded parameter.

    Returns:
        The tensor that this rank holds.
    """
    return data.to_local() if isinstance(data, DTensor) else data


def grad_like(data: DTensor, local_grad: Tensor) -> DTensor:
    """Wraps a local gradient into a ``DTensor`` with the shape, stride and placements of a parameter.

    Args:
        data: The data of the sharded parameter.
        local_grad: The local gradient of this rank.

    Returns:
        The gradient as a ``DTensor`` that views ``local_grad``.
    """
    return DTensor.from_local(
        local_grad, shape=data.shape, stride=data.stride(), device_mesh=data.device_mesh, placements=data.placements
    )


def forward_tensor_hooks(handle: ParameterHandle):
    """Runs the tensor hooks of the sharded parameter on the gradient of the unsharded one.

    Autograd computes the gradient of the unsharded parameter only, so the hooks see the local gradient of each
    backward pass, before any reduction. The hooks are read at every call, so a hook that the user removes or adds
    later takes effect at once.

    Args:
        handle: The parameter.
    """
    if not handle.unsharded.requires_grad:
        return
    sharded = handle.sharded

    def _run_hooks(grad: Tensor) -> Tensor:
        for hook in list((sharded._backward_hooks or {}).values()):  # noqa: SLF001 - PyTorch has no public reader of hooks
            result = hook(grad)
            if result is not None:
                grad = result
        return grad

    handle.unsharded.register_hook(_run_hooks)


def run_post_accumulate_hooks(handle: ParameterHandle):
    """Runs the post-accumulate-grad hooks of the sharded parameter, once its gradient is complete.

    Args:
        handle: The parameter.
    """
    # Mirrors FSDP2, which runs these hooks on the sharded parameter after the reduction.
    for hook in (handle.sharded._post_accumulate_grad_hooks or {}).values():  # noqa: SLF001 - see forward_tensor_hooks
        hook(handle.sharded)


def bind_parameters(handles: Sequence[ParameterHandle], unsharded: bool):
    """Puts unsharded or sharded parameters into every module that holds them.

    Args:
        handles: The parameters.
        unsharded: Whether to put the unsharded parameters, otherwise the sharded ones.
    """
    for handle in handles:
        for module, name in handle.owners:
            # The swap runs around every segment, so it skips the checks of nn.Module.
            module._parameters[name] = handle.unsharded if unsharded else handle.sharded  # noqa: SLF001 - see above


def _make_handle(param: nn.Parameter, compute_dtype: torch.dtype | None, training: bool) -> ParameterHandle:
    local = local_tensor(param.data)
    has_own_copy = compute_dtype is not None and compute_dtype != local.dtype
    if has_own_copy:
        # Planning needs only the shape and the dtype. A gathered parameter never needs the copy, so it is made
        # after the gathered buckets bind, see materialize_own_copies().
        source = torch.empty(local.shape, dtype=compute_dtype, device="meta")
    else:
        # The unsharded parameter views the local data, so optimizer writes reach it directly.
        source = local.detach()
    unsharded = nn.Parameter(source, requires_grad=param.requires_grad)
    # Gradients keep the precision of the storage, whatever dtype the compute weights use.
    unsharded.grad_dtype = param.grad_dtype or param.dtype
    return ParameterHandle([], param, unsharded, training and param.requires_grad, has_own_copy)


def materialize_own_copies(handles: Iterable[ParameterHandle]):
    """Allocates the compute dtype copies of the parameters without a gather.

    Args:
        handles: The parameters. A gathered parameter has no own copy and is skipped.
    """
    for handle in handles:
        if handle.has_own_copy and handle.unsharded.is_meta:
            grad_dtype = handle.unsharded.grad_dtype
            copy = local_tensor(handle.sharded.data).detach().to(handle.unsharded.dtype)
            handle.unsharded = nn.Parameter(copy, requires_grad=handle.unsharded.requires_grad)
            handle.unsharded.grad_dtype = grad_dtype


def build_handles(
    segments: Sequence[StageSegment], compute_dtype: torch.dtype | None, training: bool
) -> list[list[ParameterHandle]]:
    """Builds the parameters that each segment uses.

    A tied parameter has one handle in every segment that uses it, so autograd adds up its gradients.

    Args:
        segments: The segments of a stage.
        compute_dtype: The dtype of unsharded parameters, or ``None`` to keep the dtype of the storage.
        training: Whether the job computes gradients.

    Returns:
        The parameters of each segment, by segment index.
    """
    known: dict[int, ParameterHandle] = {}
    handles: list[list[ParameterHandle]] = []
    for segment in segments:
        segment_handles: dict[int, ParameterHandle] = {}
        for owner in segment.modules:
            for submodule in owner.modules():
                for name, param in submodule.named_parameters(recurse=False, remove_duplicate=False):
                    if id(param) not in known:
                        known[id(param)] = _make_handle(param, compute_dtype, training)
                    handle = known[id(param)]
                    handle.owners.append((submodule, name))
                    segment_handles[id(param)] = handle
        handles.append(list(segment_handles.values()))
    return handles


@torch.no_grad()
def refresh_own_copies(handles: Sequence[Sequence[ParameterHandle]]):
    """Copies the sharded weights into the unsharded parameters that have their own memory.

    Args:
        handles: The parameters of each segment.
    """
    for segment_handles in handles:
        for handle in segment_handles:
            if handle.has_own_copy:
                handle.unsharded.data.copy_(local_tensor(handle.sharded.data))


def convert_buffers(module: nn.Module, compute_dtype: torch.dtype):
    """Converts the non-persistent floating-point buffers of a module to the compute dtype.

    Persistent buffers are training state, such as the expert bias of a router, so they keep their dtype.

    Args:
        module: The module.
        compute_dtype: The dtype of the computation.
    """
    for submodule in module.modules():
        non_persistent = submodule._non_persistent_buffers_set  # noqa: SLF001 - PyTorch has no public reader
        for name, buffer in submodule.named_buffers(recurse=False):
            if buffer.is_floating_point() and name in non_persistent:
                submodule.register_buffer(name, buffer.to(compute_dtype), persistent=False)
