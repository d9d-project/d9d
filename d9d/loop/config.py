from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from d9d.pipelining.factory import AnyPipelineScheduleConfig
from d9d.tracker import AnyTrackerConfig, RunConfig


class StepActionSpecial(StrEnum):
    last_step = "last_step"
    disable = "disable"


StepActionPeriod = int | StepActionSpecial


class BatchingConfig(BaseModel):
    global_batch_size: int
    microbatch_size: int


class DeterminismConfig(BaseModel):
    base_seed: int


class PipeliningConfig(BaseModel):
    schedule: AnyPipelineScheduleConfig
    n_sequential_microbatches: int


class GarbageCollectionConfig(BaseModel):
    period_steps: StepActionPeriod


class DataLoadingConfig(BaseModel):
    num_workers: int
    pin_memory: bool
    persistent_workers: bool


class CheckpointingConfig(BaseModel):
    save_dir: Path
    period_steps: StepActionPeriod
    num_to_keep: int | None


class ModelStageFactoryConfig(BaseModel):
    source_checkpoint: Path | None
    checkpoint_only_trainable_parameters: bool


class GradientClippingConfig(BaseModel):
    max_norm: float | None
    log_total_steps: StepActionPeriod
    log_per_parameter_steps: StepActionPeriod


class ProfilingConfig(BaseModel):
    traces_dir: Path

    period_steps: int
    warmup_steps: int
    active_steps: int


class JobLoggerConfig(BaseModel):
    period_steps: StepActionPeriod


class GradientManagerConfig(BaseModel):
    grad_dtype: str | None
    bucket_size_mb: int


class TrainerConfig(BaseModel):
    run: RunConfig
    batching: BatchingConfig
    data_loading: DataLoadingConfig
    tracker: AnyTrackerConfig
    logging: JobLoggerConfig
    pipelining: PipeliningConfig | None
    model_stage_factory: ModelStageFactoryConfig
    determinism: DeterminismConfig
    gc: GarbageCollectionConfig
    checkpointing: CheckpointingConfig
    gradient_clipping: GradientClippingConfig
    profiling: ProfilingConfig | None
    gradient_manager: GradientManagerConfig


class InferenceConfig(BaseModel):
    batching: BatchingConfig
    data_loading: DataLoadingConfig
    model_stage_factory: ModelStageFactoryConfig
    determinism: DeterminismConfig
    gc: GarbageCollectionConfig
    checkpointing: CheckpointingConfig
    profiling: ProfilingConfig | None
