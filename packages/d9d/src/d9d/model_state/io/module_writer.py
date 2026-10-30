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
    mapper = ModelStateMapperSequential(
        [
            ModelStateMapperParallel(
                [_build_extraction_mapper(name, current_state_dict[name]) for name in states_to_save]
            ),
            mapper,
        ]
    )
    return mapper


def _state_generator(models: list[nn.Module]) -> Iterable[tuple[str, torch.Tensor]]:
    for model in models:
        yield from model.state_dict().items()


def save_model_state(
    dest_dir: Path, mapper: ModelStateMapper, model: nn.Module, shard_size_gb: float = 4.0, show_progress: bool = True
):
    """Saves a PyTorch module to disk from a **single** process.

    ``DTensor`` states are gathered into full tensors before saving.

    Only states listed in the inputs of ``mapper`` are saved. To save every model state unchanged, build the
    mapper with ``d9d.model_state.mapper.adapters.identity_mapper_from_module``.

    Args:
        dest_dir: The directory to save ``.safetensors`` shards and the index to.
        mapper: The mapper from model keys to on-disk keys.
        model: The PyTorch module to save.
        shard_size_gb: Maximum size of a shard file in GiB.
        show_progress: Whether to display a progress bar.
    """
    write_model_state_local(
        dest_dir=dest_dir,
        mapper=_augment_mapper_for_extraction([model], mapper),
        state_generator=_state_generator([model]),
        shard_size_gb=shard_size_gb,
        show_progress=show_progress,
    )


def save_model_state_pipeline_parallel(
    dest_dir: Path,
    mapper: ModelStateMapper,
    device_mesh: DeviceMesh,
    pipeline_dim_name: str,
    models: list[nn.Module],
    shard_size_gb: float = 4.0,
    show_progress: bool = True,
    position: int | None = None,
):
    """Saves a pipeline-parallel model to disk.

    Every rank must call this function.

    1.  **Gathering**: ``DTensor`` states are gathered into full tensors before saving.
    2.  **Single writer**: For each pipeline stage, only one rank writes files, so ranks do not overwrite each
        other.
    3.  **Index merging**: The indices of all pipeline stages are merged into one global index file.

    Only states listed in the inputs of ``mapper`` are saved. To save every model state unchanged, build the
    mapper with ``d9d.model_state.mapper.adapters.identity_mapper_from_module``.

    Args:
        dest_dir: The directory to save ``.safetensors`` shards and the index to.
        mapper: The mapper from model keys to on-disk keys.
        device_mesh: The device mesh of the whole job.
        pipeline_dim_name: The name of the pipeline-parallel dimension in ``device_mesh``.
        models: The modules (pipeline stages) held by this rank.
        shard_size_gb: Maximum size of a shard file in GiB.
        show_progress: Whether to display a progress bar.
        position: Row index for the tqdm bar. Pass the process local rank to stack one bar
            per rank without interleaving. ``None`` lets tqdm use its default (single bar).
    """
    write_model_state_pipeline_parallel(
        dest_dir=dest_dir,
        mapper=_augment_mapper_for_extraction(models, mapper),
        state_generator=_state_generator(models),
        device_mesh=device_mesh,
        pipeline_dim_name=pipeline_dim_name,
        shard_size_gb=shard_size_gb,
        show_progress=show_progress,
        position=position,
    )
