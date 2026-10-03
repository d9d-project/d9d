from collections.abc import Sequence
from collections.abc import Set as AbstractSet

import torch

from d9d.model_state.mapper.abc import ModelStateMapper, StateGroup
from d9d.model_state.mapper.compose.helper import filter_empty_mappers
from d9d.model_state.mapper.compose.parallel import ModelStateMapperParallel
from d9d.model_state.mapper.leaf.single_tensor import ModelStateMapperIdentity


class ModelStateMapperSequential(ModelStateMapper):
    """Executes a list of mappers one after another.

    Intermediate states stay hidden: the mapper exposes only the inputs of the first stage and the outputs of the
    last stage.

    1.  **Gap filling**: If a tensor must pass a stage unchanged to reach a later stage or the final output, an
        identity mapper is added for it.
    2.  **Group merging**: The mapper reports the net dependency graph. If stage A maps ``x`` to ``y`` and stage B
        maps ``y`` to ``z``, the mapper reports a single group ``{x} -> {z}``.
    """

    def __init__(self, mappers: list[ModelStateMapper]):
        """Constructs the ``ModelStateMapperSequential`` object.

        Args:
            mappers: The stages in execution order. Mappers without inputs and outputs are dropped.

        Raises:
            ValueError: If no mapper with inputs or outputs is given.
        """
        mappers = filter_empty_mappers(mappers)
        if not mappers:
            raise ValueError("Mappers list cannot be empty. Pass at least one mapper with inputs or outputs.")

        mappers = self._fill_gaps(mappers)

        self._groups = self._compute_pipeline_groups(mappers)
        self._mappers = mappers

    @staticmethod
    def _fill_gaps(mappers: list[ModelStateMapper]) -> list[ModelStateMapper]:
        mappers = mappers.copy()

        # Propagate inputs from bottom to top.
        for stage_i in range(1, len(mappers))[::-1]:
            groups_current = mappers[stage_i].state_dependency_groups()
            groups_prev = mappers[stage_i - 1].state_dependency_groups()
            current_stage_requires = frozenset.union(*(x.inputs for x in groups_current))
            prev_stage_produces = frozenset.union(*(x.outputs for x in groups_prev))

            needs_to_pass_through = current_stage_requires - prev_stage_produces

            mappers[stage_i - 1] = ModelStateMapperParallel(
                [mappers[stage_i - 1]] + [ModelStateMapperIdentity(x) for x in needs_to_pass_through]
            )

        # Propagate outputs from top to bottom.
        for stage_i in range(0, len(mappers) - 1):
            groups_current = mappers[stage_i].state_dependency_groups()
            groups_next = mappers[stage_i + 1].state_dependency_groups()
            current_stage_produces = frozenset.union(*(x.outputs for x in groups_current))
            next_stage_requires = frozenset.union(*(x.inputs for x in groups_next))

            needs_to_pass_through = current_stage_produces - next_stage_requires

            mappers[stage_i + 1] = ModelStateMapperParallel(
                [mappers[stage_i + 1]] + [ModelStateMapperIdentity(x) for x in needs_to_pass_through]
            )

        return mappers

    @staticmethod
    def _compute_pipeline_groups(mappers: list[ModelStateMapper]) -> frozenset[StateGroup]:
        outputs_depend_on_inputs = {}

        # _fill_gaps made each stage consume only keys the previous stage produces,
        # so each output group can be traced back stage by stage.
        for last_group_traced in mappers[-1].state_dependency_groups():
            required_inputs = last_group_traced.inputs

            for mapper_i in range(0, len(mappers) - 1)[::-1]:
                next_visit_groups = [
                    x for x in mappers[mapper_i].state_dependency_groups() if not x.outputs.isdisjoint(required_inputs)
                ]

                required_inputs = frozenset.union(*(x.inputs for x in next_visit_groups))

            outputs_depend_on_inputs[last_group_traced.outputs] = required_inputs

        return ModelStateMapperSequential._merge_groups(list(outputs_depend_on_inputs.items()))

    @staticmethod
    def _merge_groups(groups: Sequence[tuple[AbstractSet[str], AbstractSet[str]]]) -> frozenset[StateGroup]:
        saved_groups: list[tuple[set[str], set[str]]] = []

        saved_groups_modified = True
        while saved_groups_modified:
            saved_groups_modified = False
            for output_names, input_names in groups:
                was_new_group_created = False
                for group in saved_groups:
                    if group[0].intersection(input_names) or group[1].intersection(output_names):
                        group[0].update(input_names)
                        group[1].update(output_names)
                        was_new_group_created = True
                        saved_groups_modified = True

                if not was_new_group_created:
                    saved_groups.append((set(input_names), set(output_names)))

            groups = saved_groups
            saved_groups = []

        return frozenset(StateGroup(inputs=frozenset(x[0]), outputs=frozenset(x[1])) for x in groups)

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return self._groups

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        current_state = group
        next_state = {}
        for mapper in self._mappers:
            for deps in mapper.state_dependency_groups():
                if not deps.inputs <= current_state.keys():
                    continue

                next_state.update(mapper.apply({k: v for k, v in current_state.items() if k in deps.inputs}))

            current_state = next_state
            next_state = {}

        return current_state
