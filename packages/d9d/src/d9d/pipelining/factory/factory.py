from __future__ import annotations

import dataclasses
import typing
from collections.abc import Callable

from torch import nn
from torch.distributed import DeviceMesh

from ..api import PipelineSchedule, PipelineStageInfo
from ..api.types import TPipelineInput, TPipelineOutput, TSharedInput
from ..program.builder import (
    PipelineProgramBuilder,
    build_stage_to_host_rank_topology,
    invert_stage_to_host_rank_topology,
)
from ..program.catalogue import (
    DualPipeVPipelineProgramBuilder,
    Interleaved1F1BPipelineProgramBuilder,
    LoopedBFSPipelineProgramBuilder,
    ZeroBubbleVPipelineProgramBuilder,
)
from ..runtime import OfflinePipelineExecutor, PipelineScheduleExecutor
from ..stage import PipelineStage
from .config import (
    AnyPipelineScheduleConfig,
    PipelineSchedule1F1BConfig,
    PipelineScheduleDualPipeVConfig,
    PipelineScheduleGPipeConfig,
    PipelineScheduleInferenceConfig,
    PipelineScheduleLoopedBFSConfig,
    PipelineScheduleZeroBubbleVConfig,
)

if typing.TYPE_CHECKING:
    from d9d.parallel.runtime import ParameterRuntime


@dataclasses.dataclass(kw_only=True)
class PipelineScheduleInfo(typing.Generic[TPipelineInput, TSharedInput, TPipelineOutput]):
    """The built pipeline schedule and its rank-specific metadata.

    Attributes:
        schedule: The schedule that runs the pipeline steps.
        has_first_stage: Whether this rank hosts the first pipeline stage.
        has_last_stage: Whether this rank hosts the last pipeline stage.
    """

    schedule: PipelineSchedule[TPipelineInput, TSharedInput, TPipelineOutput]
    has_first_stage: bool
    has_last_stage: bool


def build_program_builder(config: AnyPipelineScheduleConfig) -> PipelineProgramBuilder:
    """Builds the program builder of a built-in pipeline schedule.

    Args:
        config: The configuration of the schedule.

    Returns:
        The program builder for ``build_schedule``.

    Raises:
        ValueError: If the configuration type is unknown.
    """
    match config:
        case PipelineScheduleInferenceConfig():
            return LoopedBFSPipelineProgramBuilder(num_stages_per_rank=1, inference_mode=True)
        case PipelineScheduleGPipeConfig():
            return LoopedBFSPipelineProgramBuilder(num_stages_per_rank=1, inference_mode=False)
        case PipelineScheduleLoopedBFSConfig():
            return LoopedBFSPipelineProgramBuilder(num_stages_per_rank=config.num_stages_per_rank, inference_mode=False)
        case PipelineSchedule1F1BConfig():
            return Interleaved1F1BPipelineProgramBuilder(
                num_stages_per_rank=config.num_stages_per_rank, enable_zero_bubble=config.zero_bubble
            )
        case PipelineScheduleDualPipeVConfig():
            return DualPipeVPipelineProgramBuilder()
        case PipelineScheduleZeroBubbleVConfig():
            return ZeroBubbleVPipelineProgramBuilder()
        case _:
            raise ValueError(f"Unknown pipeline schedule config ({type(config).__name__}).")


def _build_schedule_local(
    program_builder: PipelineProgramBuilder,
    model_provider: Callable[[PipelineStageInfo], nn.Module],
    runtime: ParameterRuntime | None,
) -> tuple[PipelineScheduleInfo, list[nn.Module]]:
    stage_info = PipelineStageInfo(num_stages=1, current_stage=0)

    model = model_provider(stage_info)
    scheduler = OfflinePipelineExecutor(model=model, do_backward=program_builder.has_backward, runtime=runtime)
    if runtime is not None:
        runtime.register_stage(stage_info.current_stage, model)

    return PipelineScheduleInfo(schedule=scheduler, has_first_stage=True, has_last_stage=True), [model]


def _build_schedule_distributed(
    pp_mesh: DeviceMesh,
    program_builder: PipelineProgramBuilder,
    model_provider: Callable[[PipelineStageInfo], nn.Module],
    runtime: ParameterRuntime | None,
) -> tuple[PipelineScheduleInfo, list[nn.Module]]:
    num_stages = program_builder.num_stages_per_rank * pp_mesh.size()

    stage_to_host = build_stage_to_host_rank_topology(
        num_stages=num_stages, pp_size=pp_mesh.size(), style=program_builder.topology_style
    )
    host_to_stage = invert_stage_to_host_rank_topology(stage_to_host)
    this_rank_stages = host_to_stage[pp_mesh.get_local_rank()]

    stages = []
    modules = []
    has_first_stage = False
    has_last_stage = False

    for stage_idx in this_rank_stages:
        stage_info = PipelineStageInfo(num_stages=num_stages, current_stage=stage_idx)

        if stage_info.is_current_stage_first:
            has_first_stage = True
        if stage_info.is_current_stage_last:
            has_last_stage = True

        model = model_provider(stage_info)
        modules.append(model)
        stage = PipelineStage(
            info=stage_info,
            module=model,
            group=pp_mesh.get_group(),
            stage_to_host_topology=stage_to_host,
            runtime=runtime,
        )
        stages.append(stage)

    schedule = PipelineScheduleExecutor(
        pp_group=pp_mesh.get_group(),
        stages=stages,
        program_builder=program_builder,
        runtime=runtime,
    )

    return PipelineScheduleInfo(
        schedule=schedule,
        has_first_stage=has_first_stage,
        has_last_stage=has_last_stage,
    ), modules


def build_schedule(
    pp_mesh: DeviceMesh | None,
    program_builder: PipelineProgramBuilder,
    model_provider: Callable[[PipelineStageInfo], nn.Module],
    runtime: ParameterRuntime | None = None,
) -> tuple[PipelineScheduleInfo, list[nn.Module]]:
    """Constructs the pipeline schedule and instantiates model stages.

    With ``pp_mesh``, it builds a ``PipelineScheduleExecutor`` with the stages this rank hosts.
    Without it, it builds an ``OfflinePipelineExecutor`` that runs the model directly.
    The number of microbatches is not fixed here: each step takes it from the pack it receives.

    Args:
        pp_mesh: The one-dimensional pipeline-parallel mesh, or ``None`` to run the model without
            distribution.
        program_builder: The builder of the schedule programs. Use ``build_program_builder`` for a built-in
            schedule, or pass your own ``PipelineProgramBuilder``.
        model_provider: A factory function that accepts stage info and returns the ``nn.Module``
            for that stage.
        runtime: The ``ParameterRuntime`` that owns the unsharded parameters, or ``None`` to run each stage as one
            autograd graph.

    Returns:
        A tuple of the schedule info and the list of modules created on this rank.

    Raises:
        ValueError: If ``pp_mesh`` has more than one dimension.
    """
    if pp_mesh is None:
        return _build_schedule_local(program_builder=program_builder, model_provider=model_provider, runtime=runtime)

    if pp_mesh.ndim != 1:
        raise ValueError(
            f"pp_mesh must have one dimension, but pp_mesh.ndim ({pp_mesh.ndim}). Pass the pipeline-parallel "
            'dimension only, e.g. mesh["pp"].'
        )

    return _build_schedule_distributed(
        pp_mesh=pp_mesh,
        program_builder=program_builder,
        model_provider=model_provider,
        runtime=runtime,
    )
