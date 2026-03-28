import torch

from d9d.model_state.mapper import ModelStateMapper, StateGroup


def _build_groups(mapper: ModelStateMapper, prefix: str) -> frozenset[StateGroup]:
    groups = set()
    for group in mapper.state_dependency_groups():
        scoped_inputs = frozenset(f"{prefix}{k}" for k in group.inputs)
        scoped_outputs = frozenset(f"{prefix}{k}" for k in group.outputs)
        groups.add(StateGroup(inputs=scoped_inputs, outputs=scoped_outputs))

    return frozenset(groups)


class ModelStateMapperPrefixScope(ModelStateMapper):
    """
    Encapsulates a child mapper and isolates its execution by applying a string prefix.

    This mapper allows a child mapper designed for a specific submodule
    (e.g., operating on "in_proj") to be seamlessly integrated into a larger
    parent module's state dictionary by virtually scoping its operations
    using a prefix (e.g., executing within "model.mlp.").
    """

    def __init__(self, mapper: ModelStateMapper, prefix: str) -> None:
        """
        Constructs a ModelStateMapperPrefixScope object.

        Args:
            mapper: The encapsulated child mapper to execute within the scope.
            prefix: The string prefix defining the scope boundary.
        """
        self._mapper = mapper
        self._prefix = prefix
        self._groups = _build_groups(mapper, prefix)

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return self._groups

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        scoped_group = {k.removeprefix(self._prefix): v for k, v in group.items()}

        scoped_result = self._mapper.apply(scoped_group)

        return {f"{self._prefix}{k}": v for k, v in scoped_result.items()}
