from typing import Any

import torch
from torch import nn

from d9d.pipelining.api import PipelineLossFn, PipelineResultFn, PipelineSchedule, PipelineShardingSpec


class OfflinePipelineExecutor(PipelineSchedule):
    """
    Executes the model immediately without pipeline parallelism.

    This schedule treats the execution as a single stage with a single microbatch,
    running the forward and optionally backward pass directly. This is primarily
    used for single-device execution within the pipeline abstraction.
    """

    def __init__(self, model: nn.Module, callback: PipelineLossFn | PipelineResultFn, do_backward: bool):
        """
        Constructs the offline pipeline executor.

        Args:
            model: The PyTorch module to execute.
            callback: Function to compute loss or process pipeline results.
            do_backward: Whether to execute the backward pass.
        """

        self._model = model
        self._callback = callback
        self._do_backward = do_backward

    def configure_buffers(
        self, inputs: dict[str, torch.Tensor], kwargs: dict[str, Any], sharding_spec: PipelineShardingSpec | None
    ):
        pass

    def _forward_only(self, inputs: dict[str, torch.Tensor], kwargs: dict[str, Any]):
        result = self._model(**inputs, **kwargs)
        self._callback(result, 0)  # microbatch=0

    def _forward_backward(self, inputs: dict[str, torch.Tensor], kwargs: dict[str, Any]):
        result = self._model(**inputs, **kwargs)
        loss = self._callback(result, 0)  # microbatch=0
        del result  # do not peak memory
        loss.backward()

    def step(self, inputs: dict[str, torch.Tensor], kwargs: dict[str, Any]):
        result = self._model(**inputs, **kwargs)
        processing_result = self._callback(result, 0)
        if self._do_backward:
            if not isinstance(processing_result, torch.Tensor):
                raise ValueError("Loss should be torch.Tensor")
            del result  # do not peak memory
            processing_result.backward()
