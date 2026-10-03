from typing import Generic

import torch

from d9d.pipelining.api import PipelineLossFn, PipelineResultFn, TPipelineOutput


class PipelineResultHandler(Generic[TPipelineOutput]):
    """Wraps a callback function to handle results from pipeline execution."""

    def __init__(self, callback_fn: PipelineResultFn[TPipelineOutput]):
        """Constructs the ``PipelineResultHandler`` object.

        Args:
            callback_fn: The function called with results.
        """
        self._callback_fn = callback_fn

    def trigger(self, forward_result: TPipelineOutput, microbatch_index: int):
        """Invokes the underlying callback with the provided results.

        Args:
            forward_result: The ``PipelineOutput`` produced by the last stage.
            microbatch_index: The index of the current microbatch.
        """
        self._callback_fn(forward_result, microbatch_index)


class PipelineLossHandler(Generic[TPipelineOutput]):
    """Manages loss computation and state caching across forward and backward passes."""

    def __init__(self, callback_fn: PipelineLossFn[TPipelineOutput]):
        """Constructs the ``PipelineLossHandler`` object.

        Args:
            callback_fn: The callable that computes loss from model outputs.
        """
        self._callback_fn = callback_fn
        self._cached_values: dict[int, torch.Tensor] = {}

    def trigger(self, forward_result: TPipelineOutput, microbatch_index: int):
        """Computes loss for a given microbatch result and caches it.

        Args:
            forward_result: The ``PipelineOutput`` produced by the last stage.
            microbatch_index: The index of the microbatch being processed.
        """
        result = self._callback_fn(forward_result, microbatch_index)
        self._cached_values[microbatch_index] = result

    def acquire_loss(self, microbatch_index: int) -> torch.Tensor:
        """Retrieves and releases the cached loss tensor for the backward pass.

        The loss is removed from the cache, so each loss can be acquired only once.

        Args:
            microbatch_index: The index of the microbatch.

        Returns:
            The previously computed loss tensor.

        Raises:
            ValueError: If the loss for this microbatch was not computed or was already acquired.
        """
        if microbatch_index not in self._cached_values:
            raise ValueError(
                f"No cached loss for microbatch_index ({microbatch_index}). "
                "The loss must be computed in the forward pass and acquired only once."
            )

        return self._cached_values.pop(microbatch_index)
