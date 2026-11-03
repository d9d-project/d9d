from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from torch.autograd.profiler import record_function
from torch.distributed import ProcessGroup

from d9d import pytree
from d9d.pipelining.api import PipelineLossFn, PipelineResultFn, PipelineSchedule
from d9d.pipelining.api.tensor import TensorSpec
from d9d.pipelining.stage import PipelineStage

from .action import ActionContext
from .callback import PipelineLossHandler, PipelineResultHandler
from .communications import PipelineCommunicationHandler
from .program_cache import PipelineProgramCache

if TYPE_CHECKING:
    from d9d.parallel.runtime import ParameterRuntime
    from d9d.pipelining.program.builder import PipelineProgramBuilder

_logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class _MicrobatchInputSpec:
    """A hashable shape/dtype fingerprint of one microbatch's ``PipelineInput`` tensor leaves."""

    leaf_specs: tuple[TensorSpec, ...]

    @classmethod
    def of(cls, microbatch: Any) -> _MicrobatchInputSpec:
        """Builds a fingerprint from a single microbatch's ``PipelineInput``.

        Args:
            microbatch: The microbatch's ``PipelineInput`` PyTree.

        Returns:
            A hashable fingerprint of its tensor leaf specs, in flatten (leaf) order.
        """
        return cls(
            leaf_specs=tuple(
                TensorSpec(shape=tuple(tensor.shape), dtype=tensor.dtype, layout=tensor.layout)
                for tensor in pytree.tree_leaves(microbatch)
            )
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _BufferConfig:
    """Identifies a stage-buffer allocation: the input fingerprint of every microbatch in the pack.

    Buffers are sized per microbatch, so the key covers all of them. Stage buffers are reconfigured
    when the microbatch count or any microbatch shape differs from the last step.
    """

    microbatches: tuple[_MicrobatchInputSpec, ...]

    @classmethod
    def of(cls, inputs_microbatches: tuple[Any, ...]) -> _BufferConfig:
        """Builds a buffer config from the per-microbatch inputs of a pack.

        Args:
            inputs_microbatches: The per-microbatch ``PipelineInput`` whose shapes/dtypes size buffers.

        Returns:
            A hashable buffer configuration key covering every microbatch.
        """
        return cls(microbatches=tuple(_MicrobatchInputSpec.of(microbatch) for microbatch in inputs_microbatches))


class PipelineScheduleExecutor(PipelineSchedule[Any, Any, Any]):
    """Executor that runs a pipeline schedule as its sequence of actions."""

    def __init__(
        self,
        pp_group: ProcessGroup,
        stages: list[PipelineStage],
        program_builder: PipelineProgramBuilder,
        runtime: ParameterRuntime | None = None,
    ):
        """Constructs the ``PipelineScheduleExecutor`` object.

        Args:
            pp_group: The pipeline-parallel process group.
            stages: List of stages managed by this executor.
            program_builder: Builder that composes the per-rank action program for a microbatch count.
            runtime: The ``ParameterRuntime`` that owns the gradient reductions, or ``None`` if the stages run as one
                autograd graph each.
        """
        self._stages = {stage.info.current_stage: stage for stage in stages}
        self._programs = PipelineProgramCache(pp_group, program_builder, runtime)
        self._comm_handler = PipelineCommunicationHandler(self._stages)

        self._buffer_config: _BufferConfig | None = None

    def _configure_buffers(self, inputs_microbatches: tuple[Any, ...], has_backward: bool):
        config = _BufferConfig.of(inputs_microbatches)
        if config == self._buffer_config:
            return

        for stage in self._stages.values():
            stage.configure_buffers(
                pipeline_inputs_per_microbatch=inputs_microbatches,
                has_backward=has_backward,
            )

        self._buffer_config = config

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

        expected_structure = pytree.tree_flatten(inputs_microbatches[0])[1]
        if any(pytree.tree_flatten(microbatch)[1] != expected_structure for microbatch in inputs_microbatches):
            raise ValueError("All microbatches in a pack must share the same PipelineInput structure.")

        program = self._programs.program_for(num_microbatches)
        self._configure_buffers(inputs_microbatches, program.has_backward)

        callback_fn = PipelineLossHandler(callback) if program.has_backward else PipelineResultHandler(callback)

        _logger.debug("Begin pipeline step")

        for stage in self._stages.values():
            stage.reset()

        ctx = ActionContext(
            callback=callback_fn,
            stages=self._stages,
            communications=self._comm_handler,
            pipeline_inputs_microbatches=inputs_microbatches,
            pipeline_shared_microbatches=shared_microbatches,
            reductions=program.reductions,
        )

        for action in program.program_this_rank:
            with record_function(str(action)):
                _logger.debug(f"Running pipeline action {action}")
                action.apply(ctx)

        _logger.debug("Waiting for pending PP send comms")
        self._comm_handler.wait_send_all()
        if not program.has_backward:
            # Inference has no backward pass to consume cached graphs, and sends no longer need their outputs.
            for stage in self._stages.values():
                stage.reset()
        _logger.debug("End pipeline step")
