from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from d9d.pipelining.factory import AnyPipelineScheduleConfig, PipelineScheduleGPipeConfig
from d9d.tracker import AnyTrackerConfig, RunConfig

from .types import StepActionPeriod


class JobScheduleConfig(BaseModel):
    """Configuration for the job's duration.

    Attributes:
        total_steps: The total number of steps to run. If ``None``, the length of the microbatch pack
            stream sets the duration.
    """

    model_config = ConfigDict(extra="forbid")

    total_steps: int | None = None


class DataPrefetchConfig(BaseModel):
    """Configuration for copying microbatch packs to the device ahead of the steps that consume them.

    Attributes:
        prefetch_factor: The number of packs copied to the device ahead of the current step. The copies run on
            a separate CUDA stream and overlap with compute. Each prefetched pack occupies device memory.
            ``0`` disables prefetching, so every pack is copied on the current stream when its step starts.
    """

    model_config = ConfigDict(extra="forbid")

    prefetch_factor: int = Field(default=1, ge=0)


class DeterminismConfig(BaseModel):
    """Configuration for reproducibility and random number generation.

    Attributes:
        base_seed: The base seed for the random number generators (Python, NumPy, PyTorch) on all ranks.
    """

    model_config = ConfigDict(extra="forbid")

    base_seed: int


class PipeliningConfig(BaseModel):
    """Configuration for pipeline parallelism orchestration.

    Attributes:
        schedule: The pipeline schedule configuration.
    """

    model_config = ConfigDict(extra="forbid")

    schedule: AnyPipelineScheduleConfig = Field(default_factory=PipelineScheduleGPipeConfig)


class GarbageCollectionConfig(BaseModel):
    """Configuration for manual Python garbage collection control.

    Attributes:
        period_steps: How often to run the Python garbage collector.
    """

    model_config = ConfigDict(extra="forbid")

    period_steps: StepActionPeriod


class CheckpointingConfig(BaseModel):
    """Configuration for saving model snapshots.

    Attributes:
        save_dir: The root directory for checkpoints.
        period_steps: How often to save a checkpoint.
        num_to_keep: The maximum number of recent checkpoints to keep. If ``None``, all checkpoints are kept.
    """

    model_config = ConfigDict(extra="forbid")

    save_dir: Path
    period_steps: StepActionPeriod
    num_to_keep: int | None = None


class ModelStageFactoryConfig(BaseModel):
    """Configuration for initializing model weights.

    Attributes:
        source_checkpoint: The path to a checkpoint to load into the model before the job starts. If ``None``,
            the model is initialized randomly.
        checkpoint_only_trainable_parameters: If ``True``, checkpoints store only parameters with
            ``requires_grad=True``. Useful for PEFT, e.g. LoRA.
    """

    model_config = ConfigDict(extra="forbid")

    source_checkpoint: Path | None = None
    checkpoint_only_trainable_parameters: bool = False


class GradientClippingConfig(BaseModel):
    """Configuration for gradient norm clipping.

    Attributes:
        max_norm: The maximum gradient norm. If ``None``, gradients are not clipped.
    """

    model_config = ConfigDict(extra="forbid")

    max_norm: float | None


class ProfilingConfig(BaseModel):
    """Configuration for the PyTorch Profiler.

    Attributes:
        traces_dir: The directory for trace files.
        period_steps: The total length of a profiling cycle (wait + warmup + active), in steps.
        warmup_steps: The number of profiler warmup steps before recording.
        active_steps: The number of steps to record.
        record_shapes: Whether to record the input shapes of operators.
        with_stack: Whether to record the Python call stacks of operators. They make up most of a trace.
    """

    model_config = ConfigDict(extra="forbid")

    traces_dir: Path

    period_steps: int = Field(gt=0)
    warmup_steps: int = Field(ge=0)
    active_steps: int = Field(gt=0)

    record_shapes: bool = True
    with_stack: bool = True

    @model_validator(mode="after")
    def _check_cycle_fits_period(self) -> Self:
        if self.warmup_steps + self.active_steps > self.period_steps:
            raise ValueError(
                f"Profiling period_steps ({self.period_steps}) must cover warmup_steps ({self.warmup_steps}) "
                f"plus active_steps ({self.active_steps})."
            )
        return self


class JobLoggerConfig(BaseModel):
    """Configuration for experiment tracking and logging.

    Attributes:
        period_steps: How often the metrics are logged. The metrics are accumulated over the steps between two
            logging steps. The loss and the total gradient norm are logged every step.
        tracker: The experiment tracker backend configuration, e.g. Aim.
    """

    model_config = ConfigDict(extra="forbid")

    period_steps: StepActionPeriod
    tracker: AnyTrackerConfig


class GradientManagerConfig(BaseModel):
    """Configuration for gradient synchronization.

    Attributes:
        grad_dtype: The name of the ``torch`` dtype for gradients, e.g. ``"float32"``. If ``None``, gradients
            use the parameter dtype.
        bucket_size_mb: The maximum size of a gradient communication bucket, in MiB.
    """

    model_config = ConfigDict(extra="forbid")

    grad_dtype: str | None = None
    bucket_size_mb: int = 32


class TimeoutConfig(BaseModel):
    """Configuration for distributed process group timeouts.

    Attributes:
        init_timeout: The timeout in seconds for the job setup and the first step.
        step_timeout: The timeout in seconds for communication in later steps.
    """

    model_config = ConfigDict(extra="forbid")

    init_timeout: int = 10000
    step_timeout: int = 100


class TrainerConfig(BaseModel):
    """Configuration for a complete training job.

    Attributes:
        run: Meta-information about the run (name, ID, tags).
        schedule: Job duration settings.
        data_prefetch: Settings for copying data to the device ahead of time.
        logging: Experiment tracking settings.
        pipelining: Pipeline parallelism schedule and settings.
        model_stage_factory: Model initialization and additional checkpointing logic.
        determinism: Random seed settings.
        gc: Garbage collection settings.
        checkpointing: Checkpoint saving settings.
        gradient_clipping: Gradient clipping settings.
        profiling: Profiler settings. If ``None``, profiling is disabled.
        gradient_manager: Gradient synchronization settings.
        timeout: Distributed timeout settings.
    """

    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    schedule: JobScheduleConfig = Field(default_factory=JobScheduleConfig)
    data_prefetch: DataPrefetchConfig = Field(default_factory=DataPrefetchConfig)
    logging: JobLoggerConfig
    pipelining: PipeliningConfig = Field(default_factory=PipeliningConfig)
    model_stage_factory: ModelStageFactoryConfig = Field(default_factory=ModelStageFactoryConfig)
    determinism: DeterminismConfig
    gc: GarbageCollectionConfig
    checkpointing: CheckpointingConfig
    gradient_clipping: GradientClippingConfig
    profiling: ProfilingConfig | None = None
    gradient_manager: GradientManagerConfig = Field(default_factory=GradientManagerConfig)
    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig)


class InferenceConfig(BaseModel):
    """Configuration for a complete inference or evaluation job.

    Attributes:
        schedule: Job duration settings.
        data_prefetch: Settings for copying data to the device ahead of time.
        model_stage_factory: Model initialization logic.
        determinism: Random seed settings.
        gc: Garbage collection settings.
        checkpointing: Checkpointing settings.
        profiling: Profiler settings. If ``None``, profiling is disabled.
        timeout: Distributed timeout settings.
    """

    model_config = ConfigDict(extra="forbid")

    schedule: JobScheduleConfig = Field(default_factory=JobScheduleConfig)
    data_prefetch: DataPrefetchConfig = Field(default_factory=DataPrefetchConfig)
    model_stage_factory: ModelStageFactoryConfig = Field(default_factory=ModelStageFactoryConfig)
    determinism: DeterminismConfig
    gc: GarbageCollectionConfig
    checkpointing: CheckpointingConfig
    profiling: ProfilingConfig | None = None
    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig)
