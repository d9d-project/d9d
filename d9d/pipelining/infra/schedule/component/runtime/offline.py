from typing import Any

import torch
from torch import nn

from d9d.pipelining.api import PipelineLossFn, PipelineResultFn, PipelineSchedule


class OfflinePipelineExecutor(PipelineSchedule):
    """Executes the model immediately without pipeline parallelism.

    This schedule treats the execution as a single stage, running the forward and optionally backward
    pass directly for every microbatch in the pack. This is primarily used for single-device execution
    within the pipeline abstraction.
    """

    def __init__(self, model: nn.Module, do_backward: bool):
        """Constructs the offline pipeline executor.

        Args:
            model: The PyTorch module to execute.
            do_backward: Whether to execute the backward pass.
        """
        self._model = model
        self._do_backward = do_backward

    def step(
        self,
        inputs_microbatches: tuple[dict[str, torch.Tensor], ...],
        kwargs_microbatches: tuple[dict[str, Any], ...],
        callback: PipelineLossFn | PipelineResultFn,
    ):
        num_microbatches = len(inputs_microbatches)
        if num_microbatches == 0:
            raise ValueError("Cannot run a pipeline step over an empty pack")
        if len(kwargs_microbatches) != num_microbatches:
            raise ValueError("inputs_microbatches and kwargs_microbatches must have the same length")

        for microbatch_idx in range(num_microbatches):
            inputs = inputs_microbatches[microbatch_idx]
            kwargs = kwargs_microbatches[microbatch_idx]

            result = self._model(**inputs, **kwargs)
            processing_result = callback(result, microbatch_idx)

            if self._do_backward:
                if not isinstance(processing_result, torch.Tensor):
                    raise ValueError("Loss should be torch.Tensor")
                del result  # do not peak memory
                processing_result.backward()
