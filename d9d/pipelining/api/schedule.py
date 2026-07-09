import abc
from typing import Any

import torch

from .types import PipelineLossFn, PipelineResultFn

# TODO: feature - support any PyTrees as pipeline parameters


class PipelineSchedule(abc.ABC):
    """Abstract base class defining the interface for pipeline execution schedules."""

    @abc.abstractmethod
    def step(
        self,
        inputs_microbatches: tuple[dict[str, torch.Tensor], ...],
        kwargs_microbatches: tuple[dict[str, Any], ...],
        callback: PipelineLossFn | PipelineResultFn,
    ):
        """Executes a single pipeline step over one pack of microbatches.

        The schedule receives the microbatches already split: ``inputs_microbatches[i]`` and
        ``kwargs_microbatches[i]`` are the inputs and keyword arguments of the ``i``-th microbatch.
        The number of microbatches in the step is ``len(inputs_microbatches)`` and may vary between
        steps. Program compilation and buffer allocation happen lazily inside this call, reused across
        steps when the microbatch count and shapes are unchanged.

        Args:
            inputs_microbatches: Per-microbatch input tensors (fed to the first pipeline stage).
            kwargs_microbatches: Per-microbatch keyword arguments (fed to every pipeline stage).
            callback: Function to compute loss or process pipeline results.
        """
        ...
