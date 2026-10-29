import torch

from d9d.model_state.mapper.abc import ModelStateMapper, StateGroup


class ModelStateMapperShard(ModelStateMapper):
    """Mapper that restricts another mapper to one shard of its dependency groups.

    Use it to split model loading across processes. Give each process a different ``current_shard`` so that
    each one loads only its part of the checkpoint.
    """

    def __init__(self, sub_mapper: ModelStateMapper, total_shards: int, current_shard: int):
        """Constructs the ``ModelStateMapperShard`` object.

        Args:
            sub_mapper: The mapper whose dependency groups are split.
            total_shards: The total number of shards.
            current_shard: The index of the shard this mapper handles, in ``[0, total_shards)``.
        """
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
