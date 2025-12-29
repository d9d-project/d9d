from pathlib import Path

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from d9d.model_state.io import read_model_state
from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import ModelStateMapperSequential, ModelStateMapperParallel
from d9d.model_state.mapper.leaf import ModelStateMapperDistribute, ModelStateMapperIdentity


def _build_injection_mapper(name: str, state: torch.Tensor) -> ModelStateMapper:
    if isinstance(state, DTensor):
        return ModelStateMapperDistribute(name=name, placements=state.placements, device_mesh=state.device_mesh)
    else:
        return ModelStateMapperIdentity(name)


def _augment_mapper_for_injection(model: nn.Module, mapper: ModelStateMapper) -> ModelStateMapper:
    states_to_load = {output for group in mapper.state_dependency_groups() for output in group.outputs}
    current_state_dict = model.state_dict()
    mapper = ModelStateMapperSequential([
        mapper,
        ModelStateMapperParallel([_build_injection_mapper(name, current_state_dict[name]) for name in states_to_load])
    ])
    return mapper


def load_model_state(
        src_dir: Path,
        mapper: ModelStateMapper,
        device: str,
        model: nn.Module,
        show_progress: bool = True,
):
    """
    High-level utility to stream a checkpoint directly into a PyTorch module.

    This function orchestrates the full loading lifecycle:

    1.  Topology Mapping: Uses `mapper` to rename/stack/reshape on-disk states to model states.

    2.  Automatic Distribution: If the `model` contains `DTensor`s, the loaded local tensors are automatically
        sharded/replicated to match the model's placement schema.

    3.  Streaming Read & Inject: After loading and transforming a model state, it will be injected into `model`
        using `load_state_dict(...)`.

    NOTICE: Only states specified in `mapper` will be loaded! You can use
    `d9d.model_state.mapper.adapters.identity_mapper_from_module(module)` to create a mapper that will load every
    model state without changing it.

    Args:
        src_dir: Directory containing .safetensors and index files.
        mapper: The topology defining how mapping from disk keys to model keys works.
        device: The device to load tensors onto (usually "cpu" or "cuda").
        model: The model instance to load weights into.
        show_progress: Whether to display the loading progress bar.
    """

    for state_name, state_value in read_model_state(
            src_dir=src_dir,
            mapper=_augment_mapper_for_injection(model, mapper),
            device=device,
            show_progress=show_progress
    ):
        model.load_state_dict({state_name: state_value}, strict=False)
