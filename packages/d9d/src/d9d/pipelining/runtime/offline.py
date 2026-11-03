from typing import Any

import torch
from torch import nn

from d9d.pipelining.api import PipelineComputeKind, PipelineSchedule, PipelineStepHandler


class OfflinePipelineExecutor(PipelineSchedule[Any, Any, Any]):
    """Executor that runs the model immediately without pipeline parallelism.

    This schedule treats the model as a single stage. It runs the forward pass, and optionally the
    backward pass, for every microbatch in the pack. It serves single-device runs through the pipeline
    API.
    """

    def __init__(self, model: nn.Module, do_backward: bool):
        """Constructs the ``OfflinePipelineExecutor`` object.

        Args:
            model: The PyTorch module to execute.
            do_backward: Whether to execute the backward pass.
        """
        self._model = model
        self._do_backward = do_backward

    def step(
        self,
        inputs_microbatches: tuple[Any, ...],
        shared_microbatches: tuple[Any, ...],
        handler: PipelineStepHandler[Any],
    ):
        num_microbatches = len(inputs_microbatches)
        if num_microbatches == 0:
            raise ValueError("Cannot run a pipeline step over an empty pack.")
        if len(shared_microbatches) != num_microbatches:
            raise ValueError(
                f"inputs_microbatches ({num_microbatches}) and shared_microbatches ({len(shared_microbatches)}) "
                "must have the same length."
            )

        for microbatch_idx in range(num_microbatches):
            inputs = inputs_microbatches[microbatch_idx]
            shared = shared_microbatches[microbatch_idx]

            with handler.compute_context(PipelineComputeKind.forward, 0, microbatch_idx):
                result = self._model(inputs, shared)
            processing_result = handler.process_outputs(result, microbatch_idx)

            if self._do_backward:
                if not isinstance(processing_result, torch.Tensor):
                    raise ValueError(
                        f"The result of process_outputs() ({type(processing_result).__name__}) must be a loss tensor."
                    )
                # Drop the outputs before the backward pass to lower peak memory.
                del result
                with handler.compute_context(PipelineComputeKind.backward, 0, microbatch_idx):
                    processing_result.backward()
