import abc
from typing import Any


class PipelineState(abc.ABC):
    """
    Object representing the state of a pipeline.

    This class defines the interface for accessing state variables like a dictionary,
    abstracting away whether the underlying storage is local, sharded, or global.
    """

    @abc.abstractmethod
    def __setitem__(self, key: str, value: Any):
        """
        Sets a state value for a given key.

        Args:
            key: The identifier for the state variable.
            value: The value to store.
        """

    @abc.abstractmethod
    def __getitem__(self, item: str) -> Any:
        """
        Retrieves a state value for a given key.

        Args:
            item: The identifier for the state variable.

        Returns:
            The value associated with the key.
        """

    @abc.abstractmethod
    def __contains__(self, item: str) -> bool:
        """
        Checks if a key exists in the state.

        Args:
            item: The identifier to check.

        Returns:
            True if the key exists, False otherwise.
        """
