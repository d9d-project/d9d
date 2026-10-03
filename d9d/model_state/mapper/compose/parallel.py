from collections.abc import Sequence

import torch

from d9d.model_state.mapper.abc import ModelStateMapper, StateGroup
from d9d.model_state.mapper.compose.helper import filter_empty_mappers


class ModelStateMapperParallel(ModelStateMapper):
    """Executes a list of state mappers independently of each other.

    No two mappers can consume the same input key or produce the same output key. ``apply()`` routes each
    group to the mapper that declared it.
    """

    def __init__(self, mappers: Sequence[ModelStateMapper]):
        """Constructs the ``ModelStateMapperParallel`` object.

        Args:
            mappers: The mappers to run side by side. Mappers without inputs and outputs are dropped.

        Raises:
            ValueError: If two mappers share an input key or an output key.
        """
        mappers_lst = filter_empty_mappers(mappers)

        all_groups = set()
        inputs_to_mapper = {}

        seen_inputs: set[str] = set()
        seen_outputs: set[str] = set()
        for mapper in mappers_lst:
            sub_groups = mapper.state_dependency_groups()

            for sub_group in sub_groups:
                if not seen_inputs.isdisjoint(sub_group.inputs):
                    raise ValueError(
                        f"Found colliding input keys ({sub_group.inputs}). "
                        "Each input key must be consumed by one mapper only."
                    )
                seen_inputs.update(sub_group.inputs)

                if not seen_outputs.isdisjoint(sub_group.outputs):
                    raise ValueError(
                        f"Found colliding output keys ({sub_group.outputs}). "
                        "Each output key must be produced by one mapper only."
                    )
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
            raise ValueError(
                f"Tried to run a parallel mapper with an undefined group ({sorted(group_keys)}). "
                "Pass groups exactly as returned by state_dependency_groups()."
            )

        return self._inputs_to_mapper[group_keys].apply(group)
