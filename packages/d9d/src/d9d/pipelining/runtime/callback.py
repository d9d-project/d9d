from typing import Generic

import torch

from d9d.pipelining.api import PipelineStepHandler, TPipelineOutput


class PipelineResultHandler(Generic[TPipelineOutput]):
    """Passes the results of a pipeline step without the backward pass to the step handler."""

    def __init__(self, handler: PipelineStepHandler[TPipelineOutput]):
        """Constructs the ``PipelineResultHandler`` object.

        Args:
            handler: The step handler that processes the results.
        """
        self._handler = handler

    def trigger(self, forward_result: TPipelineOutput, microbatch_index: int):
        """Invokes the underlying callback with the provided results.

        Args:
            forward_result: The ``PipelineOutput`` produced by the last stage.
            microbatch_index: The index of the current microbatch.
        """
        self._handler.process_outputs(forward_result, microbatch_index)


class PipelineLossHandler(Generic[TPipelineOutput]):
    """Manager of loss computation and state caching across forward and backward passes."""

    def __init__(self, handler: PipelineStepHandler[TPipelineOutput]):
        """Constructs the ``PipelineLossHandler`` object.

        Args:
            handler: The step handler that computes the loss from the model outputs.
        """
        self._handler = handler
        self._cached_values: dict[int, torch.Tensor] = {}

    def trigger(self, forward_result: TPipelineOutput, microbatch_index: int):
        """Computes loss for a given microbatch result and caches it.

        Args:
            forward_result: The ``PipelineOutput`` produced by the last stage.
            microbatch_index: The index of the microbatch being processed.

        Raises:
            ValueError: If the step handler returns no loss.
        """
        result = self._handler.process_outputs(forward_result, microbatch_index)
        if result is None:
            raise ValueError(f"The step handler must return a loss for microbatch_index ({microbatch_index}).")
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
