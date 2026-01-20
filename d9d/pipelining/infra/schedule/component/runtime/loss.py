from collections.abc import Callable

import torch

LossFn = Callable[[dict[str, torch.Tensor], int], torch.Tensor]


class PipelineLossHandler:
    """Manages loss computation and state caching across forward and backward passes."""

    def __init__(self, loss_fn: LossFn):
        """
        Constructs the loss handler.

        Args:
            loss_fn: The callable that computes loss from model outputs.
        """

        self._loss_fn = loss_fn
        self._cached_values: dict[int, torch.Tensor] = {}

    def compute_loss(self, forward_result: dict[str, torch.Tensor], microbatch_index: int) -> torch.Tensor:
        """
        Computes loss for a given microbatch result and caches it.

        Args:
            forward_result: The output from the last stage of the model.
            microbatch_index: The index of the microbatch being processed.

        Returns:
            The computed loss tensor.
        """

        result = self._loss_fn(forward_result, microbatch_index)
        self._cached_values[microbatch_index] = result
        return result

    def acquire_loss(self, microbatch_index: int) -> torch.Tensor:
        """
        Retrieves the cached loss tensor for the backward pass and removes it from the cache.

        Args:
            microbatch_index: The index of the microbatch.

        Returns:
            The previously computed loss tensor.

        Raises:
            ValueError: If the loss for this microbatch hasn't been computed yet.
        """

        if microbatch_index not in self._cached_values:
            raise ValueError()

        return self._cached_values[microbatch_index]
