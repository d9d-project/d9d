from typing import Any

from d9d.core.sharding import ShardingSpecLeaf

from .api import PipelineState
from .storage import PipelineStateStorage


class PipelineStateGlobal(PipelineState):
    """
    Represents the global (unsharded) view of the pipeline state.
    """

    def __init__(self, storage: PipelineStateStorage):
        """
        Constructs a PipelineStateGlobal object.

        Args:
            storage: The underlying storage backend.
        """

        self._storage = storage

    def __setitem__(self, key: str, value: Any):
        self._storage.store_global((key,), value)

    def __getitem__(self, item: str) -> Any:
        return self._storage.acquire_global((item,))

    def __contains__(self, item: str) -> bool:
        return self._storage.contains((item,))


class PipelineStateShard(PipelineState):
    """
    Represents a sharded view of the pipeline state for a specific shard ID.
    """

    def __init__(self, storage: PipelineStateStorage, current_shard: int):
        """
        Constructs a PipelineStateShard object.

        Args:
            storage: The underlying storage backend.
            current_shard: The index of the partial shard this view represents.
        """

        self._storage = storage
        self._current_shard = current_shard

    def __setitem__(self, key: str, value: Any):
        self._storage.store_shard((key,), value, self._current_shard)

    def __getitem__(self, item: str) -> Any:
        return self._storage.acquire_shard((item,), self._current_shard)

    def __contains__(self, item: str) -> bool:
        return self._storage.contains((item,))


class PipelineStateHandler:
    """
    Manages the lifecycle and access patterns of pipeline states.

    This handler initializes the underlying storage and provides specific views
    (global or sharded) into that storage.
    """

    def __init__(self, sharding_spec: dict[str, ShardingSpecLeaf], num_shards: int):
        """
        Constructs a PipelineStateHandler object.

        Args:
            sharding_spec: A definition of how specific keys should be sharded.
            num_shards: The total number of shards in the pipeline.
        """

        self._storage = PipelineStateStorage(
            sharding_spec={(k,): v for k, v in sharding_spec.items()}, num_shards=num_shards
        )

    def global_state(self) -> PipelineState:
        """
        Returns a view interface for accessing global state.

        Returns:
            A PipelineState interface that accesses the full, aggregated data.
        """

        return PipelineStateGlobal(self._storage)

    def sharded_state(self, shard_id: int) -> PipelineState:
        """
        Returns a view interface for accessing state specific to a shard ID.

        Args:
            shard_id: The index of the shard to access.

        Returns:
            A PipelineState interface that accesses partial data for the given shard.
        """

        return PipelineStateShard(self._storage, shard_id)

    def reset(self):
        """
        Resets the underlying storage, clearing all state.
        """

        self._storage.reset()
