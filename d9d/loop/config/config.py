from pathlib import Path

from pydantic import BaseModel

from d9d.pipelining.factory import AnyPipelineScheduleConfig
from d9d.tracker import AnyTrackerConfig, RunConfig

from .types import StepActionPeriod


class BatchingConfig(BaseModel):
    """
    Configuration for batch sizing logic.

    Attributes:
        global_batch_size: The total effective batch size across all distributed
            replicas and gradient accumulation steps.
        microbatch_size: The distinct batch size fed into the model during a single
            forward pass on a single device.
    """
    global_batch_size: int
    microbatch_size: int


class DeterminismConfig(BaseModel):
    """
    Configuration for reproducibility and random number generation.

    Attributes:
        base_seed: The base integer seed used to initialize random number
            generators (Python, NumPy, PyTorch) across all ranks.
    """
    base_seed: int


class PipeliningConfig(BaseModel):
    """
    Configuration for pipeline parallelism orchestration.

    Attributes:
        schedule: The specific scheduling strategy configuration used to manage pipeline execution.
    """
    schedule: AnyPipelineScheduleConfig


class GarbageCollectionConfig(BaseModel):
    """
    Configuration for manual Python garbage collection control.

    Attributes:
        period_steps: How frequently to manually trigger the Python garbage collector.
    """
    period_steps: StepActionPeriod


class DataLoadingConfig(BaseModel):
    """
    Configuration for PyTorch DataLoaders.

    Attributes:
        num_workers: The number of subprocesses to use for data loading.
        pin_memory: Whether to copy tensors into CUDA pinned memory before returning them.
        persistent_workers: If True, the data loader will not shutdown the worker processes
            after a dataset has been consumed once.
    """
    num_workers: int
    pin_memory: bool
    persistent_workers: bool


class CheckpointingConfig(BaseModel):
    """
    Configuration for saving model snapshots.

    Attributes:
        save_dir: The root directory where checkpoints will be stored.
        period_steps: How frequently to save a checkpoint.
        num_to_keep: The maximum number of recent checkpoints to retain. If None,
            all checkpoints are kept.
    """
    save_dir: Path
    period_steps: StepActionPeriod
    num_to_keep: int | None


class ModelStageFactoryConfig(BaseModel):
    """
    Configuration for initializing model weights.

    Attributes:
        source_checkpoint: Path to an initial checkpoint to load into the model
            before training starts. If None, random initialization is used.
        checkpoint_only_trainable_parameters: If True, only parameters with
            requires_grad=True will be saved in checkpoints. Useful for PEFT/LoRA.
    """
    source_checkpoint: Path | None
    checkpoint_only_trainable_parameters: bool


class GradientClippingConfig(BaseModel):
    """
    Configuration for gradient norm clipping.

    Attributes:
        max_norm: The maximum norm value for gradient clipping. If None,
            no clipping is performed.
        log_total_steps: Frequency at which to log the total gradient norm.
    """
    max_norm: float | None
    log_total_steps: StepActionPeriod


class ProfilingConfig(BaseModel):
    """
    Configuration for the PyTorch Profiler.

    Attributes:
        enabled: Whether to enable the profiler.
        traces_dir: Directory where trace files will be saved.
        period_steps: Total length of a profiling cycle (wait + warmup + active).
        warmup_steps: Number of steps to ignore before recording to allow for warming-up.
        active_steps: Number of steps to actively record traces.
    """
    enabled: bool

    traces_dir: Path

    period_steps: int
    warmup_steps: int
    active_steps: int


class JobLoggerConfig(BaseModel):
    """
    Configuration for experiment tracking and logging.

    Attributes:
        period_steps: How frequently metrics are flushed to the logger.
        tracker: Logic for the specific tracking backend (e.g., WandB, MLflow, stdout).
    """
    period_steps: StepActionPeriod
    tracker: AnyTrackerConfig


class GradientManagerConfig(BaseModel):
    """
    Configuration for gradient synchronization.

    Attributes:
        grad_dtype: The data type to use for storing the gradient. If None, follows the model's dtype.
        bucket_size_mb: The size of gradient buckets in Megabytes for communication.
    """
    grad_dtype: str | None
    bucket_size_mb: int


class TimeoutConfig(BaseModel):
    """
    Configuration for distributed process group timeouts.

    Attributes:
        init_timeout: Timeout in seconds for initializing the process group.
        step_timeout: Timeout in seconds for individual step communications.
    """
    init_timeout: int = 10000
    step_timeout: int = 100


class TrainerConfig(BaseModel):
    """
    Top-level configuration object defining a complete training job.

    Attributes:
        run: Meta-information about the run (name, ID, tags).
        batching: Batch sizing strategy.
        data_loading: DataLoader settings.
        logging: Experiment tracking settings.
        pipelining: Pipeline Parallelism schedule and settings. If None,
            pipeline parallelism is disabled.
        model_stage_factory: Model initialization and additional checkpointing logic.
        determinism: Random seed settings.
        gc: Garbage collection settings.
        checkpointing: Checkpoint saving settings.
        gradient_clipping: Gradient clipping settings.
        profiling: Profiler settings.
        gradient_manager: Gradient Synchronization Settings.
        timeout: Distributed timeout settings.
    """
    run: RunConfig
    batching: BatchingConfig
    data_loading: DataLoadingConfig
    logging: JobLoggerConfig
    pipelining: PipeliningConfig | None
    model_stage_factory: ModelStageFactoryConfig
    determinism: DeterminismConfig
    gc: GarbageCollectionConfig
    checkpointing: CheckpointingConfig
    gradient_clipping: GradientClippingConfig
    profiling: ProfilingConfig | None
    gradient_manager: GradientManagerConfig
    timeout: TimeoutConfig


class InferenceConfig(BaseModel):
    """
    Top-level configuration object defining an inference/evaluation job.

    Attributes:
        batching: Batch sizing strategy.
        data_loading: DataLoader settings.
        model_stage_factory: Model initialization logic.
        determinism: Random seed settings.
        gc: Garbage collection settings.
        checkpointing: Checkpointing settings.
        profiling: Profiler settings.
        timeout: Distributed timeout settings.
    """
    batching: BatchingConfig
    data_loading: DataLoadingConfig
    model_stage_factory: ModelStageFactoryConfig
    determinism: DeterminismConfig
    gc: GarbageCollectionConfig
    checkpointing: CheckpointingConfig
    profiling: ProfilingConfig | None
    timeout: TimeoutConfig
