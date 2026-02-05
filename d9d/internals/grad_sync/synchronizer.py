import dataclasses
from collections import defaultdict
from typing import cast

import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from .bucket import AbstractGradientBucket, LocalGradientBucket, SyncGradientBucket


def _find_reduce_mesh(data: DTensor) -> DeviceMesh | None:
    """
    Identifies the sub-mesh required for gradient reduction based on tensor placements.

    Args:
        data: The parameter tensor.

    Returns:
        The DeviceMesh subset needed for reduction, or None if no reduction is needed.
    """

    reduce_dims: set[int] = set()

    for dim_i, dim_placement in enumerate(data.placements):
        match dim_placement:
            case Replicate():
                reduce_dims.add(dim_i)
            case Shard():
                pass
            case _:
                raise ValueError(f"Unknown grad placement: {dim_placement}")

    if len(reduce_dims) == 0:
        return None

    device_mesh: DeviceMesh = data.device_mesh

    # we are sure that device mesh contain dim names so we cast(...)
    mesh_dim_names = cast(tuple[str, ...], device_mesh.mesh_dim_names)
    reduce_mesh = device_mesh[tuple(
        mesh_dim_names[dim_i] for dim_i in reduce_dims
    )]

    return reduce_mesh


@dataclasses.dataclass(frozen=True)
class _ParameterGroupMarker:
    """
    Identifier for grouping compatible parameters into buckets.
    """

    group_i: int
    reduce_mesh: DeviceMesh | None
    device: torch.device
    grad_dtype: torch.dtype | None


def _group_params_for_buckets(
        param_groups: list[list[nn.Parameter]]
) -> dict[_ParameterGroupMarker, list[nn.Parameter]]:
    """
    Sorts parameters into groups based on their synchronization requirements.

    Args:
        param_groups: List of parameter groups (from optimizer).

    Returns:
        Dictionary mapping group markers to lists of parameters.
    """

    regrouped_params = defaultdict(list)
    for param_group_i, param_group in enumerate(param_groups):
        # iterate in reverse order to maximize overlap
        for param in param_group[::-1]:
            if not param.requires_grad:
                continue

            if isinstance(param.data, DTensor):
                reduce_mesh = _find_reduce_mesh(param.data)
            else:
                reduce_mesh = None

            group = _ParameterGroupMarker(
                group_i=param_group_i,
                reduce_mesh=reduce_mesh,
                device=param.data.device,
                grad_dtype=param.grad_dtype
            )

            regrouped_params[group].append(param)

    return regrouped_params


def _make_bucket(
        require_accumulations: int,
        group_marker: _ParameterGroupMarker,
        parameters: list[nn.Parameter],
        communicate_stream: torch.cuda.Stream
) -> AbstractGradientBucket:
    """
    Factory function to create the appropriate bucket type.
    """

    if group_marker.reduce_mesh is None:
        return LocalGradientBucket(parameters)
    else:
        if group_marker.grad_dtype is None:
            raise ValueError("Gradient dtype could not be None")

        return SyncGradientBucket(
            parameters=parameters,
            require_accumulations=require_accumulations,
            device=group_marker.device,
            grad_dtype=group_marker.grad_dtype,
            reduce_mesh=group_marker.reduce_mesh,
            communicate_stream=communicate_stream
        )


def _fill_buckets(
        param_groups: dict[_ParameterGroupMarker, list[nn.Parameter]],
        bucket_size_mb: int,
        require_accumulations: int,
        communicate_stream: torch.cuda.Stream
) -> list[AbstractGradientBucket]:
    """
    Splits grouped parameters into buckets based on size constraints.

    Args:
        param_groups: Parameters grouped by sync requirements.
        bucket_size_mb: Max size for each bucket in megabytes.
        require_accumulations: Number of gradient accumulations required before syncing gradients.

    Returns:
        List of configured gradient buckets.
    """

    # TODO: Better grouping - probably we could trace autograd graph and use some topological clustering here
    # TODO: to maximize overlap even better - current implementation just iterates over parameters in reverse order
    buckets = []

    bucket_size = bucket_size_mb * 1024 * 1024

    for param_group_marker, param_group in param_groups.items():
        current_bucket_size = 0
        unfinished_bucket: list[nn.Parameter] = []
        for param in param_group:
            param_bytes = param.numel() * param.element_size()
            if current_bucket_size + param_bytes >= bucket_size and unfinished_bucket:
                buckets.append(_make_bucket(
                    require_accumulations=require_accumulations,
                    group_marker=param_group_marker,
                    parameters=unfinished_bucket,
                    communicate_stream=communicate_stream
                ))
                unfinished_bucket = []
                current_bucket_size = 0

            unfinished_bucket.append(param)
            current_bucket_size += param_bytes

        if unfinished_bucket:
            buckets.append(_make_bucket(
                require_accumulations=require_accumulations,
                group_marker=param_group_marker,
                parameters=unfinished_bucket,
                communicate_stream=communicate_stream
            ))
    return buckets


class GradientSynchronizer:
    """
    Manages gradient synchronization for distributed training.

    This class handles the bucketing of parameters, memory allocation for flat
    gradient buffers, and the orchestration of asynchronous all-reduce operations
    during the backward pass.
    """

    def __init__(
            self,
            param_groups: list[list[nn.Parameter]],
            bucket_size_mb: int,
            require_accumulations: int
    ):
        """
        Constructs a GradientSynchronizer.

        Args:
            param_groups: List of parameter groups.
            bucket_size_mb: Maximal size of a single gradient bucket in MB.
            require_accumulations: Number of micro-batches to accumulate before reducing.
        """

        self._param_groups = param_groups
        self._bucket_size_mb = bucket_size_mb
        self._require_accumulations = require_accumulations

        self._communicate_stream: torch.cuda.Stream | None = None
        self._can_sync: bool
        self._buckets: list[AbstractGradientBucket] = []

    def bind(self):
        """
        Initializes the synchronizer for training.

        Groups parameters, creates buckets, allocates memory, and registers hooks.
        Must be called before the backward pass.
        """

        stream = torch.cuda.Stream()
        self._communicate_stream = stream
        self._buckets = _fill_buckets(
            _group_params_for_buckets(self._param_groups),
            bucket_size_mb=self._bucket_size_mb,
            require_accumulations=self._require_accumulations,
            communicate_stream=stream
        )

        for bucket in self._buckets:
            bucket.bind()

    def unbind(self):
        """
        Releases resources.

        Destroys buckets, frees memory buffers, and removes hooks.
        """

        for bucket in self._buckets:
            bucket.unbind()

        self._buckets = []
        self._communicate_stream = None

    def wait(self):
        """
        Waits for all bucket operations (async reductions) to complete.
        """

        torch.cuda.current_stream().wait_stream(self._communicate_stream)

        for bucket in self._buckets:
            bucket.mark_sync()

    def zero_grad(self):
        """
        Resets gradients and accumulation counters for all managed parameters.
        """

        for bucket in self._buckets:
            bucket.zero_grad()
