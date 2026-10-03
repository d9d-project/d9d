from pathlib import Path

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import (
    ModelStateMapperParallel,
    ModelStateMapperSequential,
)
from d9d.model_state.mapper.leaf import (
    ModelStateMapperDistribute,
    ModelStateMapperIdentity,
)

from .reader import read_model_state


def _build_injection_mapper(name: str, state: torch.Tensor) -> ModelStateMapper:
    if isinstance(state, DTensor):
        return ModelStateMapperDistribute(name=name, placements=state.placements, device_mesh=state.device_mesh)
    else:
        return ModelStateMapperIdentity(name)


def _augment_mapper_for_injection(model: nn.Module, mapper: ModelStateMapper) -> ModelStateMapper:
    states_to_load = {output for group in mapper.state_dependency_groups() for output in group.outputs}
    current_state_dict = model.state_dict()
    mapper = ModelStateMapperSequential(
        [
            mapper,
            ModelStateMapperParallel(
                [_build_injection_mapper(name, current_state_dict[name]) for name in states_to_load]
            ),
        ]
    )
    return mapper


def load_model_state(
    src_dir: Path,
    mapper: ModelStateMapper,
    device: str,
    model: nn.Module,
    show_progress: bool = True,
    position: int | None = None,
):
    """Streams a checkpoint from disk into a PyTorch module.

    1.  **Mapping**: ``mapper`` renames, stacks or reshapes on-disk states into model states.
    2.  **Distribution**: if a model state is a ``DTensor``, the loaded tensor is distributed to match its
        device mesh and placements.
    3.  **Injection**: each transformed state is loaded into ``model`` with ``load_state_dict`` as soon as it is
        ready.

    Only states listed in the outputs of ``mapper`` are loaded. To load every model state unchanged, build the
    mapper with ``d9d.model_state.mapper.adapters.identity_mapper_from_module``.

    Args:
        src_dir: Directory containing the checkpoint: either sharded ``.safetensors`` files described by a
            ``model.safetensors.index.json`` file, or a single unindexed ``model.safetensors`` file.
        mapper: The mapper from on-disk keys to model keys.
        device: The device to load tensors onto, e.g. ``"cpu"`` or ``"cuda"``.
        model: The model instance to load weights into.
        show_progress: Whether to display the loading progress bar.
        position: Row index for the tqdm bar. Pass the process local rank to stack one bar
            per rank without interleaving. ``None`` lets tqdm use its default (single bar).
    """
    for state_name, state_value in read_model_state(
        src_dir=src_dir,
        mapper=_augment_mapper_for_injection(model, mapper),
        device=device,
        show_progress=show_progress,
        position=position,
    ):
        model.load_state_dict({state_name: state_value}, strict=False)
