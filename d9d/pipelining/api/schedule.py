import abc
from typing import Any

import torch

# TODO: feature - support any PyTrees as pipeline parameters


class PipelineSchedule(abc.ABC):
    """Abstract base class defining the interface for pipeline execution schedules."""

    @abc.abstractmethod
    def configure_buffers(self, inputs: dict[str, torch.Tensor], kwargs: dict[str, Any]):
        """
        Configures internal state and buffers based on input shapes.

        This method allows the schedule to pre-allocate memory or setup sharding
        specifications based on the structure of the input data before execution begins.

        Args:
            inputs: A dictionary of input tensors.
            kwargs: A dictionary of keyword arguments.
        """

        ...

    @abc.abstractmethod
    def step(self, inputs: dict[str, torch.Tensor], kwargs: dict[str, Any]):
        """
        Executes a single pipeline step using the provided inputs.

         This typically involves distributing inputs across microbatches,
         executing forward and backward passes according to the specific schedule logic,
         and handling communications between stages.

         Args:
             inputs: A dictionary of global input tensors.
             kwargs: A dictionary of global keyword arguments.
         """

        ...
