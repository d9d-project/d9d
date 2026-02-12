import dataclasses
from collections.abc import Callable

from torch import nn

from ...core.dist_context import REGULAR_DOMAIN, DistributedContext
from ..api import PipelineLossFn, PipelineResultFn, PipelineSchedule, PipelineStageInfo
from ..infra.schedule.component.program import (
    build_stage_to_host_rank_topology,
    invert_stage_to_host_rank_topology,
)
from ..infra.schedule.component.runtime import OfflinePipelineExecutor, PipelineScheduleExecutor
from ..infra.stage import PipelineStage
from .config import (
    AnyPipelineScheduleConfig,
    PipelineScheduleInferenceConfig,
)
from .registry import PIPELINE_PROGRAM_REGISTRY


@dataclasses.dataclass(kw_only=True)
class PipelineScheduleInfo:
    """Contains the built pipeline schedule and rank-specific metadata."""

    schedule: PipelineSchedule
    has_first_stage: bool
    has_last_stage: bool


def _build_schedule_local(
    schedule_config: AnyPipelineScheduleConfig,
    model_provider: Callable[[PipelineStageInfo], nn.Module],
    callback: PipelineLossFn | PipelineResultFn,
) -> tuple[PipelineScheduleInfo, list[nn.Module]]:
    stage_info = PipelineStageInfo(num_stages=1, current_stage=0)

    model = model_provider(stage_info)
    has_backward = not isinstance(schedule_config, PipelineScheduleInferenceConfig)
    scheduler = OfflinePipelineExecutor(model=model, callback=callback, do_backward=has_backward)

    return PipelineScheduleInfo(schedule=scheduler, has_first_stage=True, has_last_stage=True), [model]


def _build_schedule_distributed(
    dist_context: DistributedContext,
    n_microbatches: int,
    schedule_config: AnyPipelineScheduleConfig,
    model_provider: Callable[[PipelineStageInfo], nn.Module],
    callback: PipelineLossFn | PipelineResultFn,
) -> tuple[PipelineScheduleInfo, list[nn.Module]]:
    program_builder = PIPELINE_PROGRAM_REGISTRY.program_for(schedule_config)
    mesh = dist_context.mesh_for(REGULAR_DOMAIN)["pp"]

    num_stages = program_builder.num_stages_per_rank * mesh.size()

    stage_to_host = build_stage_to_host_rank_topology(
        num_stages=num_stages, pp_size=mesh.size(), style=program_builder.topology_style
    )
    host_to_stage = invert_stage_to_host_rank_topology(stage_to_host)
    this_rank_stages = host_to_stage[mesh.get_local_rank()]

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
            info=stage_info, module=model, group=mesh.get_group(), stage_to_host_topology=stage_to_host
        )
        stages.append(stage)

    program = program_builder.compose(num_microbatches=n_microbatches, pp_size=mesh.size())
    schedule = PipelineScheduleExecutor(
        dist_context=dist_context, stages=stages, num_microbatches=n_microbatches, callback=callback, program=program
    )

    return PipelineScheduleInfo(
        schedule=schedule, has_first_stage=has_first_stage, has_last_stage=has_last_stage
    ), modules


def build_schedule(
    dist_context: DistributedContext,
    n_microbatches: int,
    schedule_config: AnyPipelineScheduleConfig,
    model_provider: Callable[[PipelineStageInfo], nn.Module],
    callback: PipelineLossFn | PipelineResultFn,
) -> tuple[PipelineScheduleInfo, list[nn.Module]]:
    """
    Constructs the pipeline schedule and instantiates model stages.

    This function coordinates the creation of the pipeline. If the context is
    distributed, it builds a parallel schedule (`PipelineScheduleExecutor`) by
    calculating topology and creating stages for the current rank. If the
    context is local, it builds an offline schedule (`OfflinePipelineExecutor`)
    for direct execution.

    Args:
        dist_context: The distributed context.
        n_microbatches: Number of microbatches per global step.
        schedule_config: Configuration object determining the schedule strategy.
        model_provider: A factory function that accepts stage info and returns an `nn.Module`
            for that specific stage.
        callback: Callback either computing loss function (if training) or just processing pipeline outputs
            (if not training).

    Returns:
        A tuple containing the schedule info (executor and metadata) and a list
        of local PyTorch modules created for this rank.
    """

    if dist_context.mesh_params.is_distributed:
        return _build_schedule_distributed(
            dist_context=dist_context,
            n_microbatches=n_microbatches,
            schedule_config=schedule_config,
            model_provider=model_provider,
            callback=callback,
        )
    else:
        return _build_schedule_local(schedule_config=schedule_config, model_provider=model_provider, callback=callback)
