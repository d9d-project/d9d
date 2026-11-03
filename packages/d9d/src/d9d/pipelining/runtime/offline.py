from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from d9d.pipelining.api import PipelineLossFn, PipelineResultFn, PipelineSchedule, StageSegment, as_segments

from ..stage.segments import SegmentChain, backward_chains, forward_chains, validate_segments

if TYPE_CHECKING:
    from d9d.parallel.runtime import ParameterRuntime


class OfflinePipelineExecutor(PipelineSchedule[Any, Any, Any]):
    """Executor that runs the model immediately without pipeline parallelism.

    This schedule treats the model as a single stage. It runs the forward pass, and optionally the
    backward pass, for every microbatch in the pack. It serves single-device runs through the pipeline
    API.
    """

    def __init__(self, model: nn.Module, do_backward: bool, runtime: ParameterRuntime | None = None):
        """Constructs the ``OfflinePipelineExecutor`` object.

        Args:
            model: The PyTorch module to execute.
            do_backward: Whether to execute the backward pass.
            runtime: The ``ParameterRuntime`` that owns the unsharded parameters, or ``None`` to run the module as one
                autograd graph.
        """
        self._model = model
        self._do_backward = do_backward
        self._runtime = runtime
        self._source = as_segments(model)
        self._segments: Sequence[StageSegment] | None = None

    def _run_with_runtime(
        self,
        inputs: Any,
        shared: Any,
        microbatch_idx: int,
        callback: PipelineLossFn | PipelineResultFn,
        final: bool,
        runtime: ParameterRuntime,
    ):
        # The parameters exist only after the module is materialized, so this cannot run in __init__.
        if self._segments is None:
            self._segments = self._source.stage_segments()
            validate_segments(self._model, self._segments)
        chain = SegmentChain(self._source, self._segments, 0, runtime)
        chain.start_forward(inputs, shared)
        result = forward_chains([chain], runtime, 0)[0]
        processing_result = callback(result, microbatch_idx)
        if self._do_backward:
            if not isinstance(processing_result, torch.Tensor):
                raise ValueError(f"The callback result ({type(processing_result).__name__}) must be a loss tensor.")
            del result
            chain.start_backward([processing_result], None)
            backward_chains([chain], runtime, 0, runtime.reduction_plan(0, final=final))

    def step(
        self,
        inputs_microbatches: tuple[Any, ...],
        shared_microbatches: tuple[Any, ...],
        callback: PipelineLossFn | PipelineResultFn,
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

            if self._runtime is not None:
                self._run_with_runtime(
                    inputs,
                    shared,
                    microbatch_idx,
                    callback,
                    final=microbatch_idx == num_microbatches - 1,
                    runtime=self._runtime,
                )
                continue

            result = self._model(inputs, shared)
            processing_result = callback(result, microbatch_idx)

            if self._do_backward:
                if not isinstance(processing_result, torch.Tensor):
                    raise ValueError(f"The callback result ({type(processing_result).__name__}) must be a loss tensor.")
                # Drop the outputs before the backward pass to lower peak memory.
                del result
                processing_result.backward()
