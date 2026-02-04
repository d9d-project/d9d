import torch

from d9d.pipelining.api import PipelineLossFn, PipelineResultFn


class PipelineResultHandler:
    """
    Wraps a callback function to handle results from pipeline execution.
    """

    def __init__(self, callback_fn: PipelineResultFn):
        """
        Constructs PipelineResultHandler object.

        Args:
            callback_fn: The function called with results.
        """

        self._callback_fn = callback_fn

    def trigger(self, forward_result: dict[str, torch.Tensor], microbatch_index: int):
        """
        Invokes the underlying callback with the provided results.

        Args:
            forward_result: Dictionary of output tensors from the pipeline.
            microbatch_index: The index of the current micro-batch.
        """

        self._callback_fn(forward_result, microbatch_index)


class PipelineLossHandler:
    """
    Manages loss computation and state caching across forward and backward passes.
    """

    def __init__(self, loss_fn: PipelineLossFn):
        """
        Constructs the loss handler.

        Args:
            loss_fn: The callable that computes loss from model outputs.
        """

        self._loss_fn = loss_fn
        self._cached_values: dict[int, torch.Tensor] = {}

    def trigger(self, forward_result: dict[str, torch.Tensor], microbatch_index: int):
        """
        Computes loss for a given microbatch result and caches it.

        Args:
            forward_result: The output from the last stage of the model.
            microbatch_index: The index of the microbatch being processed.
        """

        result = self._loss_fn(forward_result, microbatch_index)
        self._cached_values[microbatch_index] = result

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
