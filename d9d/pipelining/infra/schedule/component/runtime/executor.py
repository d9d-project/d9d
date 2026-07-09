import dataclasses
from typing import TYPE_CHECKING, Any

import torch
from torch.autograd.profiler import record_function

from d9d.core.dist_context import DistributedContext
from d9d.core.types import TensorSpec
from d9d.pipelining.api import PipelineLossFn, PipelineResultFn, PipelineSchedule
from d9d.pipelining.infra.stage import PipelineStage

from .action import ActionContext
from .callback import PipelineLossHandler, PipelineResultHandler
from .communications import PipelineCommunicationHandler
from .program_cache import PipelineProgramCache

if TYPE_CHECKING:
    from ..program import PipelineProgramBuilder


@dataclasses.dataclass(frozen=True, slots=True)
class _MicrobatchInputSpec:
    """A hashable shape/dtype fingerprint of one microbatch's named input tensors."""

    named_specs: tuple[tuple[str, TensorSpec], ...]

    @classmethod
    def of(cls, microbatch: dict[str, torch.Tensor]) -> "_MicrobatchInputSpec":
        """Builds a fingerprint from a single microbatch's inputs.

        Args:
            microbatch: The microbatch's input tensors.

        Returns:
            A hashable fingerprint of its named tensor specs.
        """
        return cls(
            named_specs=tuple(
                (name, TensorSpec(shape=tuple(tensor.shape), dtype=tensor.dtype, layout=tensor.layout))
                for name, tensor in sorted(microbatch.items())
            )
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _BufferConfig:
    """Identifies a stage-buffer allocation: the input fingerprint of every microbatch in the pack.

    Buffers are sized per microbatch, so the cache key covers all of them — a pack whose microbatch
    shapes differ from the last step's (in count or in any individual shape) reconfigures.
    """

    microbatches: tuple[_MicrobatchInputSpec, ...]

    @classmethod
    def of(cls, inputs_microbatches: tuple[dict[str, torch.Tensor], ...]) -> "_BufferConfig":
        """Builds a buffer config from the per-microbatch inputs of a pack.

        Args:
            inputs_microbatches: The per-microbatch input tensors whose shapes/dtypes size the buffers.

        Returns:
            A hashable buffer configuration key covering every microbatch.
        """
        return cls(microbatches=tuple(_MicrobatchInputSpec.of(microbatch) for microbatch in inputs_microbatches))


class PipelineScheduleExecutor(PipelineSchedule):
    """Executes a defined pipeline schedule by interpreting a sequence of actions."""

    def __init__(
        self,
        dist_context: DistributedContext,
        stages: list[PipelineStage],
        program_builder: "PipelineProgramBuilder",
        callback: PipelineLossFn | PipelineResultFn,
    ):
        """Constructs the schedule executor.

        Args:
            dist_context: The distributed context.
            stages: List of stages managed by this executor.
            program_builder: Builder that composes the per-rank action program for a microbatch count.
            callback: Function to compute loss or process pipeline results.
        """
        self._dist_ctx = dist_context
        self._stages = {stage.info.current_stage: stage for stage in stages}
        self._programs = PipelineProgramCache(dist_context, program_builder)
        self._callback_fn = callback
        self._comm_handler = PipelineCommunicationHandler(self._stages)

        self._buffer_config: _BufferConfig | None = None

    def _configure_buffers(self, inputs_microbatches: tuple[dict[str, torch.Tensor], ...], has_backward: bool):
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
        inputs_microbatches: tuple[dict[str, torch.Tensor], ...],
        kwargs_microbatches: tuple[dict[str, Any], ...],
    ):
        num_microbatches = len(inputs_microbatches)
        if num_microbatches == 0:
            raise ValueError("Cannot run a pipeline step over an empty pack")
        if len(kwargs_microbatches) != num_microbatches:
            raise ValueError("inputs_microbatches and kwargs_microbatches must have the same length")

        expected_names = inputs_microbatches[0].keys()
        if any(microbatch.keys() != expected_names for microbatch in inputs_microbatches):
            raise ValueError("All microbatches in a pack must have the same input names")

        program = self._programs.program_for(num_microbatches)
        self._configure_buffers(inputs_microbatches, program.has_backward)

        callback = (
            PipelineLossHandler(self._callback_fn) if program.has_backward else PipelineResultHandler(self._callback_fn)
        )

        self._dist_ctx.logger.debug("Begin pipeline step")

        for stage in self._stages.values():
            stage.reset()

        for action in program.program_this_rank:
            with record_function(str(action)):
                self._dist_ctx.logger.debug(f"Running pipeline action {action}")
                action.apply(
                    ActionContext(
                        callback=callback,
                        stages=self._stages,
                        communications=self._comm_handler,
                        pipeline_inputs_microbatches=inputs_microbatches,
                        pipeline_kwargs_microbatches=kwargs_microbatches,
                    )
                )

        self._dist_ctx.logger.debug("Waiting for potentially hanging PP send comms")
        self._comm_handler.wait_send_all()  # finalize just in case
        self._dist_ctx.logger.debug("End pipeline step")
