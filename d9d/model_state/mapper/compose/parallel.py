from collections.abc import Sequence

import torch

from d9d.model_state.mapper.abc import ModelStateMapper, StateGroup
from d9d.model_state.mapper.compose.helper import filter_empty_mappers


class ModelStateMapperParallel(ModelStateMapper):
    """
    Executes a list of states mappers independently alongside each other.

    This class aggregates multiple mappers into a single logical unit.
    It enforces strict isolation between the mappers: no two mappers can
    consume the same input key (input collision) or produce the same output
    key (output collision).

    During execution (`apply`), it routes the specific subset of the input dictionary
    to the sub-mapper responsible for those keys.
    """

    def __init__(self, mappers: Sequence[ModelStateMapper]):
        mappers_lst = filter_empty_mappers(mappers)

        all_groups = set()
        inputs_to_mapper = {}

        seen_inputs: set[str] = set()
        seen_outputs: set[str] = set()
        for mapper in mappers_lst:
            sub_groups = mapper.state_dependency_groups()

            for sub_group in sub_groups:
                if not seen_inputs.isdisjoint(sub_group.inputs):
                    raise ValueError(f"Found a colliding input group: {sub_group.inputs}")
                seen_inputs.update(sub_group.inputs)

                if not seen_outputs.isdisjoint(sub_group.outputs):
                    raise ValueError(f"Found colliding output keys: {sub_group.outputs}")
                seen_outputs.update(sub_group.outputs)

                all_groups.add(sub_group)
                inputs_to_mapper[sub_group.inputs] = mapper

        self._all_groups = frozenset(all_groups)
        self._inputs_to_mapper = inputs_to_mapper

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return self._all_groups

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        group_keys = frozenset(group.keys())

        if group_keys not in self._inputs_to_mapper:
            raise ValueError("Tried to run a parallel mapper with undefined group. Perhaps you sent groups that are "
                             "not isolated?")

        return self._inputs_to_mapper[group_keys].apply(group)
