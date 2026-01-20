import dataclasses
from collections.abc import Callable
from typing import TypeVar

from torch import nn

from ...core.dist_context import REGULAR_DOMAIN, DistributedContext
from ..api import PipelineSchedule, PipelineShardingSpec, PipelineStageInfo
from ..infra.schedule.component.program import (
    PipelineProgramBuilder,
    build_stage_to_host_rank_topology,
    invert_stage_to_host_rank_topology,
)
from ..infra.schedule.component.runtime import PipelineScheduleExecutor
from ..infra.schedule.program import (
    DualPipeVPipelineProgramBuilder,
    Interleaved1F1BPipelineProgramBuilder,
    LoopedBFSPipelineProgramBuilder,
    ZeroBubbleVPipelineProgramBuilder,
)
from ..infra.stage import PipelineStage
from .config import (
    AnyPipelineScheduleConfig,
    PipelineSchedule1F1BConfig,
    PipelineScheduleDualPipeVConfig,
    PipelineScheduleGPipeConfig,
    PipelineScheduleInferenceConfig,
    PipelineScheduleLoopedBFSConfig,
    PipelineScheduleZeroBubbleVConfig,
)

TConfig = TypeVar("TConfig", bound=AnyPipelineScheduleConfig)

_CONFIG_TO_PROGRAM_REGISTRY: dict[type[TConfig], Callable[[TConfig], PipelineProgramBuilder]] = {
    PipelineScheduleGPipeConfig: (
        lambda cfg: LoopedBFSPipelineProgramBuilder(num_stages_per_rank=1, inference_mode=False)
    ),
    PipelineScheduleInferenceConfig: (
        lambda cfg: LoopedBFSPipelineProgramBuilder(num_stages_per_rank=1, inference_mode=True)
    ),
    PipelineScheduleLoopedBFSConfig: (
        lambda cfg: LoopedBFSPipelineProgramBuilder(num_stages_per_rank=cfg.num_stages_per_rank, inference_mode=False)
    ),
    PipelineSchedule1F1BConfig: (
        lambda cfg: Interleaved1F1BPipelineProgramBuilder(num_stages_per_rank=cfg.num_stages_per_rank,
                                                          enable_zero_bubble=cfg.zero_bubble)
    ),
    PipelineScheduleDualPipeVConfig: (
        lambda cfg: DualPipeVPipelineProgramBuilder()
    ),
    PipelineScheduleZeroBubbleVConfig: (
        lambda cfg: ZeroBubbleVPipelineProgramBuilder()
    )
}


@dataclasses.dataclass(kw_only=True)
class PipelineScheduleInfo:
    """Contains the built pipeline schedule and rank-specific metadata."""

    schedule: PipelineSchedule
    has_first_stage: bool
    has_last_stage: bool


def build_schedule(
        dist_context: DistributedContext,
        n_microbatches: int,
        schedule_config: AnyPipelineScheduleConfig,
        model_provider: Callable[[PipelineStageInfo], nn.Module],
        loss_fn: Callable | None,
        sharding_spec: PipelineShardingSpec
) -> tuple[PipelineScheduleInfo, list[nn.Module]]:
    """
    Constructs the pipeline schedule and instantiates model stages.

    This function coordinates the creation of the distributed pipeline. It:
    1.  Selects the appropriate `PipelineProgramBuilder` based on the config.
    2.  Calculates the global stage topology mapping stages to ranks.
    3.  Instantiates the local model stages for the current rank using `model_provider`.
    4.  Wraps models in `PipelineStage` containers.
    5.  Generates the execution program (action list).
    6.  Builds the runtime executor.

    Args:
        dist_context: The distributed context.
        n_microbatches: Number of microbatches per global step.
        schedule_config: Configuration object determining the schedule strategy.
        model_provider: A factory function that accepts stage info and returns an `nn.Module`
            for that specific stage.
        loss_fn: Optional loss function. Required if training (backward pass needed).
        sharding_spec: Specification for how data and states are sharded.

    Returns:
        A tuple containing:
        1.  `PipelineScheduleInfo`: The executable schedule and metadata.
        2.  `list[nn.Module]`: The local PyTorch modules created for this rank.
    """

    program_builder = _CONFIG_TO_PROGRAM_REGISTRY[type(schedule_config)](schedule_config)
    mesh = dist_context.mesh_for(REGULAR_DOMAIN)["pp"]

    num_stages = program_builder.num_stages_per_rank * mesh.size()

    stage_to_host = build_stage_to_host_rank_topology(
        num_stages=num_stages,
        pp_size=mesh.size(),
        style=program_builder.topology_style
    )
    host_to_stage = invert_stage_to_host_rank_topology(stage_to_host)
    this_rank_stages = host_to_stage[mesh.get_local_rank()]

    stages = []
    modules = []
    has_first_stage = False
    has_last_stage = False

    for stage_idx in this_rank_stages:
        stage_info = PipelineStageInfo(
            num_stages=num_stages,
            current_stage=stage_idx
        )

        if stage_info.is_current_stage_first:
            has_first_stage = True
        if stage_info.is_current_stage_last:
            has_last_stage = True

        model = model_provider(stage_info)
        modules.append(model)
        stage = PipelineStage(
            info=stage_info,
            module=model,
            group=mesh.get_group(),
            stage_to_host_topology=stage_to_host
        )
        stages.append(stage)

    program = program_builder.compose(num_microbatches=n_microbatches, pp_size=mesh.size())
    schedule = PipelineScheduleExecutor(
        dist_context=dist_context,
        stages=stages,
        num_microbatches=n_microbatches,
        loss_fn=loss_fn,
        program=program,
        sharding_spec=sharding_spec
    )

    return PipelineScheduleInfo(
        schedule=schedule,
        has_first_stage=has_first_stage,
        has_last_stage=has_last_stage
    ), modules
