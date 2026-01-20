import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import torch
from safetensors.torch import save_file
from torch.distributed import DeviceMesh, ProcessGroup
from tqdm import tqdm

from d9d.core.dist_ops import all_gather_object
from d9d.model_state.io.dto import (
    MODEL_STATE_INDEX_FILE_NAME,
    ModelStateIndex,
    ModelStateIndexMeta,
)
from d9d.model_state.mapper import ModelStateMapper


class _StateWritingFlowLocal:
    """
    Internal orchestration logic for buffering, transforming, and sharding model states during save.
    """

    def __init__(
            self,
            dest_dir: Path,
            mapper: ModelStateMapper,
            shard_size_gb: float,
            show_progress: bool,
            sharding_rank: int,
            # so we have to call writing flow from all processes, but
            is_current_process_rank_master: bool
    ):
        self._dest_dir = dest_dir
        self._mapper = mapper
        self._shard_size_bytes = int(shard_size_gb * (1024 ** 3))

        self._groups_to_process = set(mapper.state_dependency_groups())

        self._available_source_states: dict[str, torch.Tensor] = {}

        self._total_size = 0
        self._pending_write_tensors: dict[str, torch.Tensor] = {}
        self._current_shard_size = 0

        self._sharding_rank = sharding_rank
        self._weight_name_to_local_shard_idx: dict[str, int] = {}
        self._local_shard_idx_to_tmp_path: dict[int, Path] = {}

        self._is_current_process_rank_master = is_current_process_rank_master
        total_num_outputs = len([out_name for group in self._groups_to_process for out_name in group.outputs])
        self._pbar = tqdm(
            desc="Saving Model States",
            total=total_num_outputs,
            disable=not (show_progress and is_current_process_rank_master)
        )

    def _flush_shard(self):
        if not self._pending_write_tensors:
            return

        local_shard_num = len(self._local_shard_idx_to_tmp_path) + 1
        shard_tmp_path = self._dest_dir / f".tmp-rank{self._sharding_rank}-shard-{local_shard_num}.safetensors"

        self._local_shard_idx_to_tmp_path[local_shard_num] = shard_tmp_path
        save_file(self._pending_write_tensors, str(shard_tmp_path))

        for state_name in self._pending_write_tensors:
            self._weight_name_to_local_shard_idx[state_name] = local_shard_num

        self._pbar.update(len(self._pending_write_tensors))

        self._total_size += self._current_shard_size

        self._pending_write_tensors.clear()
        self._current_shard_size = 0

    def _process_available_groups(self):
        for group in self._groups_to_process.copy():
            if not group.inputs.issubset(self._available_source_states.keys()):
                continue

            self._groups_to_process.remove(group)

            states_to_save = self._mapper.apply(
                {k: self._available_source_states[k] for k in group.inputs}
            )

            for input_name in group.inputs:
                del self._available_source_states[input_name]

            # proceed with stateful saving only on master rank
            if self._is_current_process_rank_master:
                for name, tensor in states_to_save.items():
                    update_size = tensor.numel() * tensor.element_size()

                    if update_size > self._shard_size_bytes:
                        raise ValueError(f"Cannot save state {name} that is larger than shard size")

                    if self._current_shard_size + update_size > self._shard_size_bytes:
                        self._flush_shard()

                    self._pending_write_tensors[name] = tensor
                    self._current_shard_size += update_size

    def _finalize_locally(self) -> ModelStateIndex:
        self._flush_shard()

        if self._groups_to_process:
            missing_groups = {g.inputs for g in self._groups_to_process}
            raise ValueError(
                f"Writing failed: not all source tensors were provided to satisfy mapper dependencies. "
                f"Missing inputs for groups: {missing_groups}"
            )

        if self._available_source_states:
            warnings.warn(
                f"State Writing: The following source tensors were provided but not consumed by any "
                f"mapper group and will be ignored: {sorted(self._available_source_states.keys())}",
                stacklevel=2
            )

        weight_map_local = {
            name: self._local_shard_idx_to_tmp_path[shard_idx].name
            for name, shard_idx in self._weight_name_to_local_shard_idx.items()
        }

        return ModelStateIndex(
            metadata=ModelStateIndexMeta(total_size=self._total_size),
            weight_map=weight_map_local
        )

    def write(self, state_generator: Iterable[tuple[str, torch.Tensor]]) -> ModelStateIndex | None:
        with self._pbar:
            self._dest_dir.mkdir(parents=True, exist_ok=True)

            for name, tensor in state_generator:
                self._available_source_states[name] = tensor
                self._process_available_groups()

            if self._is_current_process_rank_master:
                return self._finalize_locally()
            else:
                return None


def _finalize_master(dest_dir: Path, indices: list[ModelStateIndex]):
    total_size = sum(index.metadata.total_size for index in indices)
    total_weight_map_local = dict(pair for index in indices for pair in index.weight_map.items())
    shard_count = len({file_name for index in indices for _, file_name in index.weight_map.items()})

    total_weight_map = {}

    local_file_to_global_file = {}
    used_global_files = 0

    for weight_name, old_file_name in total_weight_map_local.items():
        if old_file_name not in local_file_to_global_file:
            used_global_files += 1
            new_file_name = f"model-{used_global_files:05d}-of-{shard_count:05d}.safetensors"

            (dest_dir / old_file_name).rename(dest_dir / new_file_name)

            local_file_to_global_file[old_file_name] = new_file_name

        total_weight_map[weight_name] = local_file_to_global_file[old_file_name]

    index_path = dest_dir / MODEL_STATE_INDEX_FILE_NAME
    index_path.write_text(
        ModelStateIndex(
            metadata=ModelStateIndexMeta(total_size=total_size),
            weight_map=total_weight_map
        ).model_dump_json(indent=4),
        encoding="utf-8"
    )


def write_model_state_local(
        dest_dir: Path,
        mapper: ModelStateMapper,
        state_generator: Iterable[tuple[str, torch.Tensor]],
        shard_size_gb: float = 4.0,
        show_progress: bool = True
):
    """
    Saves model states to disk in a single local process.

    This function uses a streaming approach. It analyzes the mapper to determine which files
    need to be saved. Tensors are loaded into memory only when needed and evicted immediately
    after the mapper processes them.

    Args:
        dest_dir: Destination directory.
        mapper: Mapping to apply to states before saving.
        state_generator: Stream of (name, tensor) pairs to save.
        shard_size_gb: Maximum size of a single .safetensors file in GB.
        show_progress: Whether to show the progress bar.
    """
    idx = _StateWritingFlowLocal(
        dest_dir=dest_dir,
        mapper=mapper,
        shard_size_gb=shard_size_gb,
        show_progress=show_progress,
        sharding_rank=0,
        is_current_process_rank_master=True
    ).write(state_generator=state_generator)

    idx = cast(ModelStateIndex, idx)  # we are sure is_current_process_rank_master=True

    _finalize_master(dest_dir, [idx])


def write_model_state_distributed(
        dest_dir: Path,
        mapper: ModelStateMapper,
        state_generator: Iterable[tuple[str, torch.Tensor]],
        process_group: ProcessGroup,
        shard_size_gb: float = 4.0,
        show_progress: bool = True
):
    """
    Saves model states in a distributed setup (multiple processes).

    This function uses a streaming approach. It analyzes the mapper to determine which files
    need to be saved. Tensors are loaded into memory only when needed and evicted immediately
    after the mapper processes them.

    Each rank writes its own shard. Rank 0 gathers indices and finalizes the checkpoint.

    Args:
        dest_dir: Destination directory.
        mapper: Mapping to apply to states before saving.
        state_generator: Stream of (name, tensor) pairs from the model.
        process_group: The distributed process group.
        shard_size_gb: Maximum shard size in GB.
        show_progress: Whether to show the progress bar.
    """

    current_idx = _StateWritingFlowLocal(
        dest_dir=dest_dir,
        mapper=mapper,
        shard_size_gb=shard_size_gb,
        show_progress=show_progress,
        sharding_rank=process_group.rank(),
        is_current_process_rank_master=True
    ).write(state_generator=state_generator)
    gather_idx = all_gather_object(current_idx, process_group)
    gather_idx_filter = [x for x in gather_idx if x is not None]
    if process_group.rank() == 0:
        _finalize_master(dest_dir, gather_idx_filter)


def write_model_state_pipeline_parallel(
        dest_dir: Path,
        mapper: ModelStateMapper,
        state_generator: Iterable[tuple[str, torch.Tensor]],
        device_mesh: DeviceMesh,
        pipeline_dim_name: str,
        shard_size_gb: float = 4.0,
        show_progress: bool = True
):
    """
    Saves model states in a complex ND distributed training setting.

    This function uses a streaming approach. It analyzes the mapper to determine which files
    need to be saved. Tensors are loaded into memory only when needed and evicted immediately
    after the mapper processes them.

    This handles Pipeline Parallelism by ensuring that only one rank per pipeline stage
    actually writes data to disk to avoid duplication.

    Args:
        dest_dir: Destination directory.
        mapper: Mapping to apply to states before saving.
        state_generator: Stream of (name, tensor) pairs from the model.
        device_mesh: The PyTorch DeviceMesh representing the cluster layout.
        pipeline_dim_name: The name of the mesh dimension responsible for pipeline parallelism.
        shard_size_gb: Maximum shard size in GB.
        show_progress: Whether to show the progress bar.
    """

    pipeline_rank = device_mesh[pipeline_dim_name].get_rank()

    mesh_dim_names = device_mesh.mesh_dim_names
    coords = device_mesh.get_coordinate()
    if mesh_dim_names is None or coords is None:
        raise ValueError("Cannot save state using a DeviceMesh with no dim names or coords")

    non_pipeline_coord_sum = sum(
        coord
        for name, coord
        in zip(mesh_dim_names, coords, strict=True)
        if name != pipeline_dim_name
    )
    master_within_pipeline_rank = non_pipeline_coord_sum == 0

    current_idx = _StateWritingFlowLocal(
        dest_dir=dest_dir,
        mapper=mapper,
        shard_size_gb=shard_size_gb,
        show_progress=show_progress,
        sharding_rank=pipeline_rank,
        is_current_process_rank_master=master_within_pipeline_rank
    ).write(state_generator=state_generator)
    gather_idx = all_gather_object(current_idx, device_mesh.get_group(0))
    gather_idx_filter = [x for x in gather_idx if x is not None]
    if pipeline_rank == 0 and master_within_pipeline_rank:
        _finalize_master(dest_dir, gather_idx_filter)
