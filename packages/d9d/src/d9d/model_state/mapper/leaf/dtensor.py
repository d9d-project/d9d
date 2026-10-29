from collections.abc import Sequence

import torch
from torch._C._distributed import Placement
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_tensor

from d9d.model_state.mapper.abc import ModelStateMapper, StateGroup


class ModelStateMapperDistribute(ModelStateMapper):
    """Mapper that converts a single full local tensor into a ``DTensor``.

    Every rank must hold the same full tensor. No communication happens.
    """

    def __init__(self, name: str, device_mesh: DeviceMesh | None, placements: Sequence[Placement] | None):
        """Constructs the ``ModelStateMapperDistribute`` object.

        Args:
            name: The name of the tensor to distribute.
            device_mesh: The device mesh of the resulting ``DTensor``. Passed to ``distribute_tensor``.
            placements: The placements of the resulting ``DTensor``. Passed to ``distribute_tensor``.
        """
        self._name = name

        self._device_mesh = device_mesh
        self._placements = placements

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([StateGroup(inputs=frozenset([self._name]), outputs=frozenset([self._name]))])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            self._name: distribute_tensor(
                group[self._name],
                device_mesh=self._device_mesh,
                placements=self._placements,
                # Every rank already holds the full tensor, so skip the broadcast from a source rank.
                src_data_rank=None,
            )
        }


class ModelStateMapperGatherFullTensor(ModelStateMapper):
    """Mapper that gathers a single ``DTensor`` into a full local tensor."""

    def __init__(self, name: str):
        """Constructs the ``ModelStateMapperGatherFullTensor`` object.

        Args:
            name: The name of the tensor to gather.
        """
        self._name = name

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([StateGroup(inputs=frozenset([self._name]), outputs=frozenset([self._name]))])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tensor = group[self._name]

        if not isinstance(tensor, DTensor):
            raise ValueError(f"Tensor ({self._name}) type ({type(tensor).__name__}) must be DTensor to be gathered.")

        return {self._name: tensor.full_tensor()}
