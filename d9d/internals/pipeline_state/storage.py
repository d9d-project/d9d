import copy
from collections import UserDict
from typing import Any, TypeVar, cast

import torch
import torch.utils._pytree as pytree  # noqa: PLC2701

from d9d.core.sharding import ShardingSpecLeaf, SpecReplicate, SpecShard, shard_tree, unshard_tree

StateKey = tuple[str, ...]


TMap = TypeVar("TMap")


def _detach_leaf(x: TMap) -> TMap:
    """
    Detaches a tensor from the computation graph if the input is a tensor.

    Args:
        x: The input object.

    Returns:
        The detached tensor or original object.
    """

    if isinstance(x, torch.Tensor):
        return cast(TMap, x.detach())
    return x


class ShardedState(UserDict):
    """
    Container for holding state broken down by shard indices.
    """


class PipelineStateStorage:
    """
    Low-level storage backend handling sharding and aggregation of state data.

    This class manages the transition between sharded data
    and global data. It uses sharding specifications to determine
    how to split or join data.
    """

    def __init__(
            self,
            sharding_spec: dict[StateKey, ShardingSpecLeaf],
            num_shards: int
    ):
        """
        Constructs a PipelineStateStorage object.

        Args:
            sharding_spec: Dictionary mapping state keys to their sharding behaviors.
            num_shards: Total number of shards involved in the storage.
        """

        self._sharding_spec_orig = copy.deepcopy(sharding_spec)

        self._state: dict[StateKey, Any] = {}
        self._state_sharding_spec: dict[StateKey, ShardingSpecLeaf] = {}

        self._num_shards = num_shards

    def _guess_sharding_spec_for_shard(self, key: StateKey, shard: Any) -> ShardingSpecLeaf:
        # Stack if scalar (tensor or item), cat otherwise

        if key in self._sharding_spec_orig:
            return self._sharding_spec_orig[key]

        if isinstance(shard, torch.Tensor):
            do_stack = shard.ndim == 0
            return SpecShard(dim=0, do_stack=do_stack)
        elif isinstance(shard, list):
            return SpecShard(dim=0)
        else:
            return SpecShard(dim=0, do_stack=True)

    def _guess_sharding_spec_for_global(self, key: StateKey, state: Any) -> ShardingSpecLeaf:
        if key in self._sharding_spec_orig:
            return self._sharding_spec_orig[key]

        if isinstance(state, torch.Tensor):
            if state.ndim == 0:
                return SpecReplicate()
            else:
                return SpecShard(dim=0)
        elif isinstance(state, list):
            return SpecShard(dim=0)
        else:
            return SpecReplicate()

    def store_global(self, key: StateKey, state: Any):
        """
        Stores a value in the global scope.

        If the key does not have a sharding spec, one will be inferred. Detaches tensors.

        Args:
            key: The identifier key.
            state: The unified value to store.
        """

        state = pytree.tree_map(_detach_leaf, state)

        if key not in self._state_sharding_spec:
            self._state_sharding_spec[key] = self._guess_sharding_spec_for_global(key, state)

        self._state[key] = state

    def store_shard(self, key: StateKey, state: Any, shard_id: int):
        """
        Stores a value for a specific shard index.

        Raises error if attempting to shard an already global key without conversion.

        Args:
            key: The identifier key.
            state: The partial value for the shard.
            shard_id: The index of the shard.

        Raises:
            ValueError: If trying to store sharded state into an unsharded container.
        """

        if key not in self._state:
            self._state[key] = ShardedState()

        container = self._state[key]

        if not isinstance(container, ShardedState):
            raise ValueError(f"Trying to store sharded state into an unsharded one: {key}")

        state = pytree.tree_map(_detach_leaf, state)

        # dynamically populate sharding spec to know whether it is stacking or not
        if key not in self._state_sharding_spec:
            self._state_sharding_spec[key] = self._guess_sharding_spec_for_shard(key, state)

        self._state[key][shard_id] = state

    def _ensure_global(self, key: StateKey):
        if key not in self._state:
            raise ValueError(f"Cannot access non-existing state {key}")

        state = self._state[key]

        if not isinstance(state, ShardedState):
            return

        # here we know we are in ShardedState

        shards = [state[shard_id] for shard_id in range(self._num_shards)]
        resharded = unshard_tree(shards, self._state_sharding_spec[key])

        self._state[key] = resharded

    def _ensure_sharded(self, key: StateKey):
        if key not in self._state:
            raise ValueError(f"Cannot access non-existing state {key}")

        state = self._state[key]

        if isinstance(state, ShardedState):
            return

        # here we know we are in global state

        sharded = shard_tree(
            state,
            self._state_sharding_spec[key],
            num_shards=self._num_shards,
            enforce_even_split=True
        )

        sharded_state = ShardedState({
            shard_idx: shard for shard_idx, shard in enumerate(sharded)
        })

        self._state[key] = sharded_state

    def acquire_global(self, key: StateKey) -> Any:
        """
        Retrieves data for a key in its global form.

        Args:
            key: The state key.

        Returns:
            The aggregated global data.
        """

        self._ensure_global(key)
        return self._state[key]

    def acquire_shard(self, key: StateKey, shard: int) -> Any:
        """
        Retrieves data for a key specific to a shard index.

        Args:
            key: The state key.
            shard: The shard index.

        Returns:
            The data slice corresponding to the shard.
        """

        self._ensure_sharded(key)
        state = self._state[key]

        if isinstance(state, ShardedState):
            return state[shard]
        else:
            return state

    def contains(self, key: StateKey) -> bool:
        """
        Checks if a key exists in storage.

        Args:
            key: The state key.

        Returns:
            True if present.
        """

        return key in self._state

    def reset(self):
        """
        Clears all stored state.
        """

        self._state.clear()
