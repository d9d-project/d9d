import torch

from d9d.model_state.mapper.abc import ModelStateMapper, StateGroup


class ModelStateMapperShard(ModelStateMapper):
    """
    Wraps another state mapper and restricts its execution to a specific subset (shard)
    of dependency groups.

    This is primarily used for parallelizing model loading across multiple processes
    or nodes. By assigning a different `current_shard` index to each process,
    the total set of tensors required by the `sub_mapper` is split evenly,
    preventing every process from loading the entire checkpoint.
    """

    def __init__(self, sub_mapper: ModelStateMapper, total_shards: int, current_shard: int):
        self._groups = self._shard_groups(
            sub_mapper.state_dependency_groups(), n_shards=total_shards, shard=current_shard
        )
        self._sub_mapper = sub_mapper
        self._total_shards = total_shards
        self._current_shard = current_shard

    @staticmethod
    def _shard_groups(groups: frozenset[StateGroup], n_shards: int, shard: int) -> frozenset[StateGroup]:
        groups_sorted = sorted(groups, key=lambda x: sorted(x.inputs))
        groups_shard = [x for i, x in enumerate(groups_sorted) if i % n_shards == shard]
        return frozenset(groups_shard)

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return self._groups

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._sub_mapper.apply(group)
