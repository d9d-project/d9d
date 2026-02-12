from collections.abc import Callable
from typing import TypeVar, cast

from d9d.pipelining.factory import (
    AnyPipelineScheduleConfig,
    PipelineSchedule1F1BConfig,
    PipelineScheduleDualPipeVConfig,
    PipelineScheduleGPipeConfig,
    PipelineScheduleInferenceConfig,
    PipelineScheduleLoopedBFSConfig,
    PipelineScheduleZeroBubbleVConfig,
)
from d9d.pipelining.infra.schedule.component.program import PipelineProgramBuilder
from d9d.pipelining.infra.schedule.program import (
    DualPipeVPipelineProgramBuilder,
    Interleaved1F1BPipelineProgramBuilder,
    LoopedBFSPipelineProgramBuilder,
    ZeroBubbleVPipelineProgramBuilder,
)

TConfig = TypeVar("TConfig", bound=AnyPipelineScheduleConfig)

TRegistryDict = dict[type[AnyPipelineScheduleConfig], Callable[[AnyPipelineScheduleConfig], PipelineProgramBuilder]]

TBoundRegistryFn = Callable[[TConfig], PipelineProgramBuilder]


class PipelineProgramRegistry:
    def __init__(self) -> None:
        self._registry: TRegistryDict = {}

    def register_program(self, config_cls: type[TConfig]) -> Callable[[TBoundRegistryFn], TBoundRegistryFn]:
        def decorator(func: TBoundRegistryFn) -> TBoundRegistryFn:
            config_cls_any = cast(type[AnyPipelineScheduleConfig], config_cls)
            self._registry[config_cls_any] = func
            return func

        return decorator

    def program_for(self, config: AnyPipelineScheduleConfig) -> PipelineProgramBuilder:
        program_fn = self._registry[type(config)]
        program = program_fn(config)
        return program


PIPELINE_PROGRAM_REGISTRY = PipelineProgramRegistry()


@PIPELINE_PROGRAM_REGISTRY.register_program(PipelineScheduleGPipeConfig)
def _build_gpipe(_: PipelineScheduleGPipeConfig) -> PipelineProgramBuilder:
    return LoopedBFSPipelineProgramBuilder(num_stages_per_rank=1, inference_mode=False)


@PIPELINE_PROGRAM_REGISTRY.register_program(PipelineScheduleInferenceConfig)
def _build_inference(_: PipelineScheduleInferenceConfig) -> PipelineProgramBuilder:
    return LoopedBFSPipelineProgramBuilder(num_stages_per_rank=1, inference_mode=True)


@PIPELINE_PROGRAM_REGISTRY.register_program(PipelineScheduleLoopedBFSConfig)
def _build_looped_bfs(cfg: PipelineScheduleLoopedBFSConfig) -> PipelineProgramBuilder:
    return LoopedBFSPipelineProgramBuilder(num_stages_per_rank=cfg.num_stages_per_rank, inference_mode=False)


@PIPELINE_PROGRAM_REGISTRY.register_program(PipelineSchedule1F1BConfig)
def _build_1f1b(cfg: PipelineSchedule1F1BConfig) -> PipelineProgramBuilder:
    return Interleaved1F1BPipelineProgramBuilder(
        num_stages_per_rank=cfg.num_stages_per_rank, enable_zero_bubble=cfg.zero_bubble
    )


@PIPELINE_PROGRAM_REGISTRY.register_program(PipelineScheduleDualPipeVConfig)
def _build_dual_pipe_v(_: PipelineScheduleDualPipeVConfig) -> PipelineProgramBuilder:
    return DualPipeVPipelineProgramBuilder()


@PIPELINE_PROGRAM_REGISTRY.register_program(PipelineScheduleZeroBubbleVConfig)
def _build_zero_bubble_v(_: PipelineScheduleZeroBubbleVConfig) -> PipelineProgramBuilder:
    return ZeroBubbleVPipelineProgramBuilder()
