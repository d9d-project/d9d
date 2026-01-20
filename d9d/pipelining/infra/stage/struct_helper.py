from collections.abc import Iterable, Sequence
from typing import TypeVar

T = TypeVar("T")


class DictFlattener:
    """
    Helper class to flatten and unflatten dictionaries into sequences deterministically.
    """

    def __init__(self, keys: Iterable[str]):
        """
        Constructs a DictFlattener object.

        Args:
            keys: The collection of dictionary keys to manage. They will be sorted internally.
        """

        self._order_to_key = {i: x for i, x in enumerate(sorted(keys))}

    def flatten(self, inputs: dict[str, T]) -> list[T]:
        """
        Converts a dictionary into a list based on the sorted internal key order.

        Args:
            inputs: The dictionary to flatten. Must contain all keys provided at init.

        Returns:
            A list of values sorted by their corresponding keys.
        """

        return [inputs[self._order_to_key[i]] for i in range(len(inputs))]

    def unflatten(self, outputs: Sequence[T]) -> dict[str, T]:
        """
        Reconstructs a dictionary from a sequence of values.

        Args:
            outputs: A sequence of values corresponding to the sorted internal key order.

        Returns:
            A dictionary mapping original keys to the provided values.
        """

        return {self._order_to_key[i]: out for i, out in enumerate(outputs)}
