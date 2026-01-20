from collections.abc import Sequence

import torch
from torch._C._distributed import Placement
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_tensor

from d9d.model_state.mapper.abc import ModelStateMapper, StateGroup


class ModelStateMapperDistribute(ModelStateMapper):
    """
    Converts a single local Tensor object into a DTensor object with specified
    `device_mesh` and `placements`.
    """

    def __init__(self, name: str, device_mesh: DeviceMesh | None, placements: Sequence[Placement] | None):
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
                src_data_rank=None  # do not communicate here
            )
        }


class ModelStateMapperGatherFullTensor(ModelStateMapper):
    """
    Gathers a single DTensor object into a full Tensor object.
    """

    def __init__(self, name: str):
        self._name = name

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([StateGroup(inputs=frozenset([self._name]), outputs=frozenset([self._name]))])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tensor = group[self._name]

        if not isinstance(tensor, DTensor):
            raise ValueError("Cannot gather anything but DTensor")

        return {
            self._name: tensor.full_tensor()
        }
