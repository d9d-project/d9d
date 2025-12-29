from collections import defaultdict
from pathlib import Path
from typing import Generator, Iterable

import torch
from safetensors import safe_open
from tqdm import tqdm

from d9d.model_state.io.dto import ModelStateIndex, MODEL_STATE_INDEX_FILE_NAME
from d9d.model_state.mapper import ModelStateMapper


class _StateLoadingFlow:
    """
    Internal orchestration logic for loading and transforming model states in a streamed manner.
    """

    def __init__(
            self,
            src_dir: Path,
            mapper: ModelStateMapper,
            device: str,
            show_progress: bool
    ):
        self._src_dir = src_dir
        self._mapper = mapper
        self._device = device

        # I/O in constructor!
        self._index = self._load_index()
        self._groups_to_process = set(mapper.state_dependency_groups())

        self._stored_states = {}

        self._check_index()

        self._pbar = tqdm(
            desc='Loading Model States',
            total=len([output_name for group in self._groups_to_process for output_name in group.outputs]),
            disable=not show_progress
        )

    def _load_index(self) -> ModelStateIndex:
        index_file = self._src_dir / MODEL_STATE_INDEX_FILE_NAME
        index_data = index_file.read_text(encoding='utf-8')
        index = ModelStateIndex.model_validate_json(index_data)
        return index

    def _check_index(self):
        will_process_inputs = set()
        for group in self._groups_to_process:
            will_process_inputs.update(group.inputs)

        on_disk_inputs = set(self._index.weight_map.keys())

        missing_inputs = will_process_inputs.difference(on_disk_inputs)

        if len(missing_inputs) > 0:
            raise ValueError(f"Cannot run state loading: states {missing_inputs} are missing!")

    def _update_in_memory_states(self, file_to_load: str, params_to_load: set[str]):
        with safe_open(str(self._src_dir / file_to_load), framework="pt", device=str(self._device)) as st:
            for param_to_load in params_to_load:
                self._stored_states[param_to_load] = st.get_tensor(param_to_load)

    def _process_available_groups(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        for group in self._groups_to_process.copy():
            if not group.inputs.issubset(self._stored_states.keys()):
                continue

            self._groups_to_process.remove(group)

            loaded_states = self._mapper.apply(
                {k: v for k, v in self._stored_states.items() if k in group.inputs}
            )
            for state_name, state_value in loaded_states.items():
                yield state_name, state_value
            self._pbar.update(len(loaded_states))

            for input_name in group.inputs:
                del self._stored_states[input_name]

    def _build_file_loading_plan(self) -> dict[str, set[str]]:
        plan = defaultdict(set)
        for group in self._mapper.state_dependency_groups():
            for key in group.inputs:
                require_file = self._index.weight_map[key]
                plan[require_file].add(key)
        return plan

    def load(self) -> Iterable[tuple[str, torch.Tensor]]:
        with self._pbar:
            for file_to_load, params_to_load in self._build_file_loading_plan().items():
                self._update_in_memory_states(file_to_load, params_to_load)
                yield from self._process_available_groups()


def read_model_state(
        src_dir: Path,
        mapper: ModelStateMapper,
        device: str,
        show_progress: bool = True
) -> Iterable[tuple[str, torch.Tensor]]:
    """
    Reads a model checkpoint from disk, transforming it on-the-fly according to the state mapper.

    This function uses a streaming approach. It analyzes the mapper to determine which files
    need to be loaded. Tensors are loaded into memory only when needed and evicted immediately
    after the mapper processes them.

    Args:
        src_dir: The directory containing .safetensors files and `model.safetensors.index.json` file.
        mapper: The transformation graph defining how to map on-disk keys to output keys.
        device: The device to load tensors onto (e.g., "cpu", "cuda:0").
        show_progress: Whether to display a progress bar.

    Yields:
        A tuple containing the transformed parameter name and its tensor value.
    """

    yield from _StateLoadingFlow(
        src_dir=src_dir,
        device=device,
        mapper=mapper,
        show_progress=show_progress
    ).load()
