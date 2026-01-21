from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor

from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import (
    ModelStateMapperParallel,
    ModelStateMapperSequential,
)
from d9d.model_state.mapper.leaf import (
    ModelStateMapperGatherFullTensor,
    ModelStateMapperIdentity,
)

from .writer import (
    write_model_state_local,
    write_model_state_pipeline_parallel,
)


def _build_extraction_mapper(name: str, state: torch.Tensor) -> ModelStateMapper:
    if isinstance(state, DTensor):
        return ModelStateMapperGatherFullTensor(name)
    else:
        return ModelStateMapperIdentity(name)


def _augment_mapper_for_extraction(models: list[nn.Module], mapper: ModelStateMapper) -> ModelStateMapper:
    states_to_save = {input_state for group in mapper.state_dependency_groups() for input_state in group.inputs}

    current_state_dict = {}
    for model in models:
        current_state_dict.update(model.state_dict())
    mapper = ModelStateMapperSequential([
        ModelStateMapperParallel([_build_extraction_mapper(name, current_state_dict[name]) for name in states_to_save]),
        mapper
    ])
    return mapper


def _state_generator(models: list[nn.Module]) -> Iterable[tuple[str, torch.Tensor]]:
    for model in models:
        yield from model.state_dict().items()


def save_model_state(
        dest_dir: Path,
        mapper: ModelStateMapper,
        model: nn.Module,
        shard_size_gb: float = 4.0,
        show_progress: bool = True
):
    """
    High-level utility to save a PyTorch model to disk on a **single** process.

    NOTICE: Only states specified in `mapper` will be saved! You can use
    `d9d.model_state.mapper.adapters.identity_mapper_from_module(module)` to create a mapper that will save every
    model state without changing it.

    Args:
        dest_dir: The directory to save .safetensors shards and index.
        mapper: Topology defining how model keys map to disk keys.
        model: The PyTorch module to save.
        shard_size_gb: Max size per shard file in Gigabytes.
        show_progress: Whether to display a progress bar.
    """

    write_model_state_local(
        dest_dir=dest_dir,
        mapper=_augment_mapper_for_extraction([model], mapper),
        state_generator=_state_generator([model]),
        shard_size_gb=shard_size_gb,
        show_progress=show_progress
    )


def save_model_state_pipeline_parallel(
        dest_dir: Path,
        mapper: ModelStateMapper,
        device_mesh: DeviceMesh,
        pipeline_dim_name: str,
        models: list[nn.Module],
        shard_size_gb: float = 4.0,
        show_progress: bool = True
):
    """
    High-level utility to save a model in a Distributed Pipeline Parallel environment to disk.

    Features:

    1. **Auto-Gather**: Converts `DTensor` parameters to full tensors before saving.

    2. **Distribution Awareness**: Uses the `device_mesh` to ensure that for a given pipeline stage,
       only the master rank writes the checkpoint, preventing Write-After-Write conflicts.

    3. **Index Merging**: Aggregates metadata from all independent pipeline stages into one global index file.

    NOTICE: Only states specified in `mapper` will be saved! You can use
    `d9d.model_state.mapper.adapters.identity_mapper_from_module(module)` to create a mapper that will save every
    model state without changing it.

    Args:
        dest_dir: directory to save .safetensors shards and index file.
        mapper: Topology defining how model keys map to disk keys.
        device_mesh: The cluster topology mesh.
        pipeline_dim_name: The specific dimension name in the mesh used for pipelining.
        models: A list of modules (pipeline stages) processed by this PP rank.
        shard_size_gb: Max size per shard file in Gigabytes.
        show_progress: Whether to display a progress bar.
    """
    write_model_state_pipeline_parallel(
        dest_dir=dest_dir,
        mapper=_augment_mapper_for_extraction(models, mapper),
        state_generator=_state_generator(models),
        device_mesh=device_mesh,
        pipeline_dim_name=pipeline_dim_name,
        shard_size_gb=shard_size_gb,
        show_progress=show_progress
    )
