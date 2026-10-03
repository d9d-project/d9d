import dataclasses
from collections import defaultdict
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


def _make_bucket(
    group_marker: _ParameterGroupMarker,
    parameters: list[nn.Parameter],
    communicate_stream: torch.cuda.Stream,
) -> AbstractGradientBucket:
    """Creates a local bucket or a sync bucket, depending on the reduce mesh.

    Returns:
        The created bucket.

    Raises:
        ValueError: If the gradient dtype is ``None`` for a sync bucket.
    """
    if group_marker.reduce_mesh is None:
        return LocalGradientBucket(parameters)
    else:
        if group_marker.grad_dtype is None:
            raise ValueError("Gradient dtype cannot be None for parameters that need gradient synchronization.")

        return SyncGradientBucket(
            parameters=parameters,
            device=group_marker.device,
            grad_dtype=group_marker.grad_dtype,
            reduce_mesh=group_marker.reduce_mesh,
            communicate_stream=communicate_stream,
        )


def _fill_buckets(
    param_groups: dict[_ParameterGroupMarker, list[nn.Parameter]],
    bucket_size_mb: int,
    communicate_stream: torch.cuda.Stream,
) -> list[AbstractGradientBucket]:
    """Splits grouped parameters into buckets of limited size.

    Args:
        param_groups: Parameters grouped by synchronization requirements.
        bucket_size_mb: The maximum size of one bucket in MiB.
        communicate_stream: The CUDA stream for asynchronous gradient communication.

    Returns:
        The gradient buckets.
    """
    buckets = []

    bucket_size = bucket_size_mb * 1024 * 1024

    for param_group_marker, param_group in param_groups.items():
        current_bucket_size = 0
        unfinished_bucket: list[nn.Parameter] = []
        for param in param_group:
            param_bytes = param.numel() * param.element_size()
            if current_bucket_size + param_bytes >= bucket_size and unfinished_bucket:
                buckets.append(
                    _make_bucket(
                        group_marker=param_group_marker,
                        parameters=unfinished_bucket,
                        communicate_stream=communicate_stream,
                    )
                )
                unfinished_bucket = []
                current_bucket_size = 0

            unfinished_bucket.append(param)
            current_bucket_size += param_bytes

        if unfinished_bucket:
            buckets.append(
                _make_bucket(
                    group_marker=param_group_marker,
                    parameters=unfinished_bucket,
                    communicate_stream=communicate_stream,
                )
            )
    return buckets


class GradientSynchronizer:
    """Synchronizes the gradients of replicated parameters during the backward pass.

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

    def bind(self):
        """Builds the buckets, allocates their gradient buffers and registers the backward hooks.

        It must be called before the backward pass.
        """
        stream = torch.cuda.Stream()
        self._communicate_stream = stream
        self._buckets = _fill_buckets(
            _group_params_for_buckets(self._param_groups),
            bucket_size_mb=self._bucket_size_mb,
            communicate_stream=stream,
        )

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
        self._communicate_stream = None

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

    def zero_grad(self):
        """Resets gradients and accumulation counters for all managed parameters."""
        for bucket in self._buckets:
            bucket.zero_grad()

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
