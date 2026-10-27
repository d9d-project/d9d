import dataclasses
from collections import defaultdict
from collections.abc import Callable
from typing import cast

import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from .bucket import AbstractGradientBucket, LocalGradientBucket, SyncGradientBucket


def _find_reduce_mesh(data: DTensor) -> DeviceMesh | None:
    """Finds the sub-mesh to reduce the gradient over, from the placements of the parameter.

    Args:
        data: The parameter tensor.

    Returns:
        The sub-mesh of the replicated dimensions, or ``None`` if no reduction is needed.

    Raises:
        ValueError: If a placement is neither ``Replicate`` nor ``Shard``.
    """
    reduce_dims: set[int] = set()

    for dim_i, dim_placement in enumerate(data.placements):
        match dim_placement:
            case Replicate():
                reduce_dims.add(dim_i)
            case Shard():
                pass
            case _:
                raise ValueError(
                    f"Gradient placement ({dim_placement}) is not supported. Use Replicate or Shard placements."
                )

    if len(reduce_dims) == 0:
        return None

    device_mesh: DeviceMesh = data.device_mesh

    # d9d builds every device mesh with dimension names.
    mesh_dim_names = cast(tuple[str, ...], device_mesh.mesh_dim_names)
    reduce_mesh = device_mesh[tuple(mesh_dim_names[dim_i] for dim_i in reduce_dims)]

    return reduce_mesh


@dataclasses.dataclass(frozen=True)
class _ParameterGroupMarker:
    """Key that groups the parameters which can share a bucket."""

    group_i: int
    reduce_mesh: DeviceMesh | None
    device: torch.device
    grad_dtype: torch.dtype | None


def _group_params_for_buckets(
    param_groups: list[list[nn.Parameter]],
) -> dict[_ParameterGroupMarker, list[nn.Parameter]]:
    """Sorts parameters into groups by their synchronization requirements.

    Args:
        param_groups: The parameter groups, usually from the optimizer.

    Returns:
        A dict that maps group markers to their parameters.
    """
    regrouped_params = defaultdict(list)
    for param_group_i, param_group in enumerate(param_groups):
        # The backward pass produces gradients roughly in reverse parameter order, so reversed buckets
        # fill up and start their reduction earlier.
        for param in param_group[::-1]:
            if not param.requires_grad:
                continue

            if isinstance(param.data, DTensor):
                reduce_mesh = _find_reduce_mesh(param.data)
            else:
                reduce_mesh = None

            group = _ParameterGroupMarker(
                group_i=param_group_i, reduce_mesh=reduce_mesh, device=param.data.device, grad_dtype=param.grad_dtype
            )

            regrouped_params[group].append(param)

    return regrouped_params


def _split_by_size(
    params: list[nn.Parameter], grad_dtype: torch.dtype, bucket_size_mb: int
) -> list[list[nn.Parameter]]:
    """Splits parameters into consecutive chunks whose local gradients fit into ``bucket_size_mb``.

    A parameter larger than the limit gets a chunk of its own.

    Args:
        params: The ``DTensor`` parameters to split.
        grad_dtype: The dtype of the gradients.
        bucket_size_mb: The maximum size of one chunk in MiB.

    Returns:
        The chunks of parameters, in the original order.
    """
    max_bytes = bucket_size_mb * 1024 * 1024
    chunks: list[list[nn.Parameter]] = []
    chunk_bytes = 0
    for param in params:
        param_bytes = cast(DTensor, param.data).to_local().numel() * grad_dtype.itemsize
        if not chunks or chunk_bytes + param_bytes > max_bytes:
            chunks.append([])
            chunk_bytes = 0
        chunks[-1].append(param)
        chunk_bytes += param_bytes
    return chunks


def _make_buckets(
    group_marker: _ParameterGroupMarker,
    params: list[nn.Parameter],
    bucket_size_mb: int,
    communicate_stream: torch.cuda.Stream,
    on_ready: Callable[[], None],
) -> list[AbstractGradientBucket]:
    """Creates the buckets of one parameter group.

    A group without a reduce mesh becomes one local bucket. Other groups are split into sync buckets of limited size.

    Args:
        group_marker: The synchronization requirements of the group.
        params: The parameters of the group.
        bucket_size_mb: The maximum size of one sync bucket in MiB.
        communicate_stream: The CUDA stream for asynchronous gradient communication.
        on_ready: Called when a sync bucket reaches the required accumulation count.

    Returns:
        The buckets of the group.

    Raises:
        ValueError: If the gradient dtype is ``None`` for parameters that need gradient synchronization.
    """
    reduce_mesh = group_marker.reduce_mesh
    if reduce_mesh is None:
        return [LocalGradientBucket(params)]

    grad_dtype = group_marker.grad_dtype
    if grad_dtype is None:
        raise ValueError("Gradient dtype cannot be None for parameters that need gradient synchronization.")

    return [
        SyncGradientBucket(
            parameters=chunk,
            device=group_marker.device,
            grad_dtype=grad_dtype,
            reduce_mesh=reduce_mesh,
            communicate_stream=communicate_stream,
            on_ready=on_ready,
        )
        for chunk in _split_by_size(params, grad_dtype=grad_dtype, bucket_size_mb=bucket_size_mb)
    ]


class GradientSynchronizer:
    """Gradient synchronizer for replicated parameters. It reduces the gradients during the backward pass.

    It splits the parameters into buckets, allocates flat gradient buffers and runs asynchronous all-reduce
    operations.
    """

    def __init__(self, param_groups: list[list[nn.Parameter]], bucket_size_mb: int):
        """Constructs the ``GradientSynchronizer`` object.

        Args:
            param_groups: The parameter groups.
            bucket_size_mb: The maximum size of one gradient bucket in MiB.
        """
        self._param_groups = param_groups
        self._bucket_size_mb = bucket_size_mb
        self._require_accumulations: int | None = None

        self._communicate_stream: torch.cuda.Stream | None = None
        self._can_sync: bool
        self._buckets: list[AbstractGradientBucket] = []
        self._sync_buckets: list[SyncGradientBucket] = []
        self._next_reduction = 0

    def bind(self):
        """Builds the buckets, allocates their gradient buffers and registers the backward hooks.

        It must be called before the backward pass.
        """
        stream = torch.cuda.Stream()
        self._communicate_stream = stream
        self._buckets = [
            bucket
            for group_marker, params in _group_params_for_buckets(self._param_groups).items()
            for bucket in _make_buckets(
                group_marker,
                params,
                bucket_size_mb=self._bucket_size_mb,
                communicate_stream=stream,
                on_ready=self._start_ready_reductions,
            )
        ]
        self._sync_buckets = [bucket for bucket in self._buckets if isinstance(bucket, SyncGradientBucket)]
        self._next_reduction = 0

        for bucket in self._buckets:
            bucket.bind()

        # A rebind in the middle of training (e.g. after offload and onload) keeps the count of the current step.
        if self._require_accumulations is not None:
            for bucket in self._buckets:
                bucket.set_required_accumulations(self._require_accumulations)

    def unbind(self):
        """Destroys the buckets, frees their buffers and removes the hooks."""
        for bucket in self._buckets:
            bucket.unbind()

        self._buckets = []
        self._sync_buckets = []
        self._communicate_stream = None

    def _start_ready_reductions(self):
        """Starts the reductions of ready buckets in bucket order.

        A ready bucket waits until all buckets before it are ready.
        """
        # Collectives are matched by call order, not by buffer. The backward pass can make buckets ready in a
        # different order on each rank, so every rank must start them in the same bucket order.
        while (
            self._next_reduction < len(self._sync_buckets)
            and self._sync_buckets[self._next_reduction].is_ready_to_reduce()
        ):
            self._sync_buckets[self._next_reduction].start_reduction()
            self._next_reduction += 1

    def wait(self):
        """Makes the current stream wait for all asynchronous reductions and marks the buckets as synchronized.

        Raises:
            ValueError: If the synchronizer is not bound.
        """
        stream = self._communicate_stream
        if stream is None:
            raise ValueError("The synchronizer is not bound. Call bind() first.")

        torch.cuda.current_stream().wait_stream(stream)

        for bucket in self._buckets:
            bucket.mark_sync()

    @torch.no_grad()
    def scale_grads(self, factor: torch.Tensor):
        """Multiplies the gradients of all managed parameters in place.

        Args:
            factor: The scalar to multiply the gradients by.

        Raises:
            ValueError: If the synchronizer is not bound.
        """
        if self._communicate_stream is None:
            raise ValueError("The synchronizer is not bound. Call bind() first.")

        grads = [grad for bucket in self._buckets for grad in bucket.grad_local_tensors()]
        if len(grads) > 0:
            torch._foreach_mul_(grads, factor)

    def zero_grad(self):
        """Resets gradients and accumulation counters for all managed parameters."""
        for bucket in self._buckets:
            bucket.zero_grad()
        self._next_reduction = 0

    def set_required_accumulations(self, require_accumulations: int):
        """Sets the accumulation count of the current step for all buckets.

        The count can change from step to step, because the pack length varies. Later binds, for example after
        offload and onload, reuse the latest count.

        Args:
            require_accumulations: The number of accumulations required before the gradients are reduced.
        """
        self._require_accumulations = require_accumulations
        for bucket in self._buckets:
            bucket.set_required_accumulations(require_accumulations)
