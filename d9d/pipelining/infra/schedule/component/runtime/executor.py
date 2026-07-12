import dataclasses
from typing import TYPE_CHECKING, Any

from torch.autograd.profiler import record_function

from d9d.core import pytree
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
    """A hashable shape/dtype fingerprint of one microbatch's ``PipelineInput`` tensor leaves."""

    leaf_specs: tuple[TensorSpec, ...]

    @classmethod
    def of(cls, microbatch: Any) -> "_MicrobatchInputSpec":
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

    Buffers are sized per microbatch, so the cache key covers all of them — a pack whose microbatch
    shapes differ from the last step's (in count or in any individual shape) reconfigures.
    """

    microbatches: tuple[_MicrobatchInputSpec, ...]

    @classmethod
    def of(cls, inputs_microbatches: tuple[Any, ...]) -> "_BufferConfig":
        """Builds a buffer config from the per-microbatch inputs of a pack.

        Args:
            inputs_microbatches: The per-microbatch ``PipelineInput`` whose shapes/dtypes size buffers.

        Returns:
            A hashable buffer configuration key covering every microbatch.
        """
        return cls(microbatches=tuple(_MicrobatchInputSpec.of(microbatch) for microbatch in inputs_microbatches))


class PipelineScheduleExecutor(PipelineSchedule[Any, Any, Any]):
    """Executes a defined pipeline schedule by interpreting a sequence of actions."""

    def __init__(
        self,
        dist_context: DistributedContext,
        stages: list[PipelineStage],
        program_builder: "PipelineProgramBuilder",
    ):
        """Constructs the schedule executor.

        Args:
            dist_context: The distributed context.
            stages: List of stages managed by this executor.
            program_builder: Builder that composes the per-rank action program for a microbatch count.
        """
        self._dist_ctx = dist_context
        self._stages = {stage.info.current_stage: stage for stage in stages}
        self._programs = PipelineProgramCache(dist_context, program_builder)
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
            raise ValueError("Cannot run a pipeline step over an empty pack")
        if len(shared_microbatches) != num_microbatches:
            raise ValueError("inputs_microbatches and shared_microbatches must have the same length")

        expected_structure = pytree.tree_flatten(inputs_microbatches[0])[1]
        if any(pytree.tree_flatten(microbatch)[1] != expected_structure for microbatch in inputs_microbatches):
            raise ValueError("All microbatches in a pack must share the same PipelineInput structure")

        program = self._programs.program_for(num_microbatches)
        self._configure_buffers(inputs_microbatches, program.has_backward)

        callback_fn = PipelineLossHandler(callback) if program.has_backward else PipelineResultHandler(callback)

        self._dist_ctx.logger.debug("Begin pipeline step")

        for stage in self._stages.values():
            stage.reset()

        ctx = ActionContext(
            callback=callback_fn,
            stages=self._stages,
            communications=self._comm_handler,
            pipeline_inputs_microbatches=inputs_microbatches,
            pipeline_shared_microbatches=shared_microbatches,
        )

        for action in program.program_this_rank:
            with record_function(str(action)):
                self._dist_ctx.logger.debug(f"Running pipeline action {action}")
                action.apply(ctx)

        self._dist_ctx.logger.debug("Waiting for pending PP send comms")
        self._comm_handler.wait_send_all()
        self._dist_ctx.logger.debug("End pipeline step")
