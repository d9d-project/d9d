from collections.abc import Iterable
from pathlib import Path

from tqdm import tqdm

from d9d.core.dist_context import DeviceMeshParameters
from d9d.core.offload import DEFAULT_SLEEP_TAGS, SleepTag
from d9d.internals.determinism import set_seeds
from d9d.loop.component import (
    DataParallelMicrobatchPackStream,
    GradientClipper,
    GradientManager,
    JobLogger,
    JobProfiler,
    JobSchedule,
    ManualGarbageCollector,
    ModelStageExporter,
    ModelStageFactory,
    OptimizerFactory,
    PipelineStateHandler,
    StateCheckpointer,
    TimeoutManager,
    TrainSleeper,
    TrainTaskOperator,
)
from d9d.loop.config import TrainerConfig
from d9d.loop.control import (
    CreateMetricsContext,
    DataProvider,
    FinalizeContext,
    InitializeDataProviderContext,
    LRSchedulerProvider,
    ModelProvider,
    OptimizerProvider,
    RegisterModelEventsContext,
    RegisterTaskEventsContext,
    TrainTaskProvider,
    TrainTaskProviderContext,
)
from d9d.loop.event import EventBus
from d9d.loop.event.catalogue.common import (
    EventConfigurationStartedContext,
    EventDataStreamReadyContext,
    EventModelStagesReadyContext,
    EventStepContext,
)
from d9d.loop.event.catalogue.train import (
    EVENT_TRAIN_CONFIG_STARTED,
    EVENT_TRAIN_DATA_STREAM_READY,
    EVENT_TRAIN_FINISHED,
    EVENT_TRAIN_FORWARD_BACKWARD_POST,
    EVENT_TRAIN_FORWARD_BACKWARD_PRE,
    EVENT_TRAIN_LR_SCHEDULER_READY,
    EVENT_TRAIN_MODEL_STAGES_READY,
    EVENT_TRAIN_OPTIMIZER_READY,
    EVENT_TRAIN_OPTIMIZER_STEP_POST,
    EVENT_TRAIN_OPTIMIZER_STEP_PRE,
    EVENT_TRAIN_READY,
    EVENT_TRAIN_STEP_POST,
    EVENT_TRAIN_STEP_PRE,
    EventLRSchedulerReadyContext,
    EventOptimizerReadyContext,
    EventTrainFinishedContext,
    EventTrainReadyContext,
)
from d9d.loop.state import TrainJobState
from d9d.metric.impl.container import ComposeMetric

from ._device import move_pack_to_device


class TrainingConfigurator:
    """Orchestrates the assembly of the distributed training environment.

    This class binds the infrastructure configuration (DeviceMesh), the training
    parameters (TrainerConfig), and the user-defined logic (Providers) to create
    a fully initialized state object capable of running the training loop.
    """

    def __init__(
        self,
        mesh: DeviceMeshParameters,
        parameters: TrainerConfig,
        task_provider: TrainTaskProvider,
        model_provider: ModelProvider,
        data_provider: DataProvider,
        optimizer_provider: OptimizerProvider,
        lr_scheduler_provider: LRSchedulerProvider,
    ):
        """Constructs a configurator capable of building the full training state.

        Args:
            mesh: Definition of the distributed device mesh topology.
            parameters: The global configuration object for the trainer.
            task_provider: Factory for creating the training task logic.
            model_provider: Factory for defining and creating model stages.
            data_provider: Factory for building the microbatch pack stream.
            optimizer_provider: Factory for creating the optimizer.
            lr_scheduler_provider: Factory for creating the learning rate scheduler.
        """
        self._mesh = mesh
        self._parameters = parameters
        self._task_provider = task_provider
        self._model_provider = model_provider
        self._data_provider = data_provider
        self._optimizer_provider = optimizer_provider
        self._lr_scheduler_provider = lr_scheduler_provider

    def _build_new_training_state(self) -> TrainJobState:
        dist_context = self._mesh.build()

        set_seeds(dist_context, seed=self._parameters.determinism.base_seed)

        timeout_manager = TimeoutManager(dist_context=dist_context, config=self._parameters.timeout)
        timeout_manager.set_init()

        task = self._task_provider(TrainTaskProviderContext(dist_context=dist_context))

        event_bus = EventBus()
        self._model_provider.register_events(RegisterModelEventsContext(dist_context=dist_context, event_bus=event_bus))
        task.register_events(RegisterTaskEventsContext(dist_context=dist_context, event_bus=event_bus))

        event_bus.trigger(EVENT_TRAIN_CONFIG_STARTED, EventConfigurationStartedContext(dist_context=dist_context))

        microbatch_pack_stream = self._data_provider(InitializeDataProviderContext(dist_context=dist_context))
        event_bus.trigger(EVENT_TRAIN_DATA_STREAM_READY, EventDataStreamReadyContext(stream=microbatch_pack_stream))

        checkpointable_stream = DataParallelMicrobatchPackStream(
            dist_context=dist_context,
            inner=microbatch_pack_stream,
        )

        schedule = JobSchedule(
            config=self._parameters.schedule,
            stream=checkpointable_stream,
        )

        pipeline_state_handler = PipelineStateHandler()

        pipeline_schedule, modules = ModelStageFactory(
            model_provider=self._model_provider,
            dist_context=dist_context,
            config_model=self._parameters.model_stage_factory,
            config_pipelining=self._parameters.pipelining,
        ).build_pipeline_and_modules()
        event_bus.trigger(EVENT_TRAIN_MODEL_STAGES_READY, EventModelStagesReadyContext(modules=modules.modules))

        metrics = ComposeMetric(task.create_metrics(CreateMetricsContext()).metrics)

        gradient_manager = GradientManager(
            dist_context=dist_context,
            tracked_modules=modules,
            config=self._parameters.gradient_manager,
        )

        task_operator = TrainTaskOperator(
            dist_context=dist_context,
            task=task,
            pipeline=pipeline_schedule,
            pipeline_state=pipeline_state_handler,
            gradient_manager=gradient_manager,
            job_schedule=schedule,
            metrics=metrics,
        )

        grad_clipper = GradientClipper(
            dist_context=dist_context,
            tracked_modules=modules,
            config=self._parameters.gradient_clipping,
            schedule=schedule,
        )

        optimizer, scheduler = OptimizerFactory(
            dist_context=dist_context,
            tracked_modules=modules,
            optimizer_provider=self._optimizer_provider,
            schedule=schedule,
            lr_scheduler_provider=self._lr_scheduler_provider,
        ).build_optimizer_and_scheduler()
        event_bus.trigger(EVENT_TRAIN_OPTIMIZER_READY, EventOptimizerReadyContext(optimizer=optimizer))
        event_bus.trigger(EVENT_TRAIN_LR_SCHEDULER_READY, EventLRSchedulerReadyContext(lr_scheduler=scheduler))

        gc = ManualGarbageCollector(dist_ctx=dist_context, config=self._parameters.gc, schedule=schedule)

        checkpointer = StateCheckpointer(
            dist_context=dist_context,
            schedule=schedule,
            config=self._parameters.checkpointing,
            gc=gc,
            run_name=self._parameters.run.name,
        )

        profiler = JobProfiler(dist_context=dist_context, schedule=schedule, config=self._parameters.profiling)

        exporter = ModelStageExporter(model_provider=self._model_provider, dist_context=dist_context, modules=modules)

        job_logger = JobLogger(
            dist_context=dist_context,
            config=self._parameters.logging,
            metrics=metrics,
            schedule=schedule,
            run_config=self._parameters.run,
            additional_hparams={"task": task.dump_hparams(), "model": self._model_provider.dump_hparams()},
        )

        return TrainJobState(
            dist_context=dist_context,
            microbatch_pack_stream=checkpointable_stream,
            schedule=schedule,
            tracked_modules=modules,
            garbage_collector=gc,
            checkpointer=checkpointer,
            optimizer=optimizer,
            task=task,
            lr_scheduler=scheduler,
            gradient_clipper=grad_clipper,
            profiler=profiler,
            exporter=exporter,
            metrics=metrics,
            logger=job_logger,
            gradient_manager=gradient_manager,
            timeout_manager=timeout_manager,
            task_operator=task_operator,
            event_bus=event_bus,
        )

    def configure(self) -> "Trainer":
        """Instantiates all training components and returns a configured Trainer.

        This method triggers the creation of the distributed context, sets seeds,
        builds the model, optimizer, data loaders, and attaches all auxiliary
        components (logging, profiling, checkpointing).

        Returns:
            Trainer: A ready-to-use trainer instance encapsulating the job state.
        """
        state = self._build_new_training_state()

        return Trainer(state)


class Trainer:
    """The main execution engine for running a distributed training job.

    This class manages the training loop, lifecycle events, distributed synchronization,
    and periodic side-effects (logging, checkpointing).
    """

    def __init__(self, state: TrainJobState):
        """Constructs a Trainer from a pre-built job state.

        Args:
            state: The encapsulated state object containing all initialized
                components (model, optimizer, dist_context, etc.).
        """
        self._state = state
        self._sleeper = TrainSleeper(
            dist_context=state.dist_context,
            tracked_modules=state.tracked_modules,
            optimizer=state.optimizer,
            gradient_manager=state.gradient_manager,
            event_bus=state.event_bus,
        )

    def train(self):
        """Executes the full training workflow."""
        self._state.dist_context.wait_world()
        self._state.dist_context.logger.info("Trying to load last checkpoint before doing anything else")
        self._state.checkpointer.load_last_checkpoint(self._state)

        if self._state.schedule.current_step >= self._state.schedule.total_steps:
            self._state.dist_context.logger.info("Already trained fully, will do nothing")
            return

        self._state.dist_context.wait_world()

        step_ctx = EventStepContext(schedule=self._state.schedule)

        with (
            tqdm(
                desc="Training",
                total=self._state.schedule.total_steps,
                disable=not self._state.dist_context.is_local_main_process,
                initial=self._state.schedule.current_step,
            ) as bar,
            self._state.logger.new_run() as run,
            self._state.garbage_collector as gc,
            self._state.profiler.open() as profiler,
            self._state.gradient_manager.install(),
            self._state.gradient_clipper.install(),
            self._state.logger.install(),
        ):
            run.set_context({"stage": "train"})
            self._state.event_bus.trigger(EVENT_TRAIN_READY, EventTrainReadyContext(run=run))

            for pack in self._state.microbatch_pack_stream:
                run.set_step(self._state.schedule.current_step)
                self._state.event_bus.trigger(EVENT_TRAIN_STEP_PRE, step_ctx)

                device_pack = move_pack_to_device(pack, "cuda")

                with self._state.event_bus.bounded(
                    EVENT_TRAIN_FORWARD_BACKWARD_PRE, EVENT_TRAIN_FORWARD_BACKWARD_POST, step_ctx
                ):
                    # we do both forward and backward passes over the whole pack of microbatches;
                    # since GradientManager is installed - it should start performing
                    # synchronization overlapping grad sync with compute. Loss/weight is accumulated
                    # into the gradient manager per microbatch inside the loss callback.
                    self._state.task_operator.forward_backward(device_pack)

                # metrics were successfully accumulated during forward passes - we can schedule their synchronization
                self._state.logger.trigger_sync()

                # wait for gradient synchronization finishes and scale them
                self._state.gradient_manager.sync_and_scale()

                # clip grads after they are synced across world
                self._state.gradient_clipper.clip_and_log(run)

                # optimize (it won't sync grads - they are already Replicate-d)
                with self._state.event_bus.bounded(
                    EVENT_TRAIN_OPTIMIZER_STEP_PRE, EVENT_TRAIN_OPTIMIZER_STEP_POST, step_ctx
                ):
                    self._state.optimizer.step()

                # update LR
                self._state.lr_scheduler.step()

                # log everything
                self._state.logger.log(run, loss_value=self._state.gradient_manager.compute_global_loss())

                # reset grads
                self._state.gradient_manager.zero_grad()

                gc.collect_periodic()

                if profiler:
                    profiler.step()

                self._state.timeout_manager.set_periodic()

                self._state.event_bus.trigger(EVENT_TRAIN_STEP_POST, step_ctx)
                self._state.schedule.step()

                # checkpoint at the end of the step
                self._state.checkpointer.checkpoint_if_needed(self._state)

                bar.update()

            self._state.task.finalize(FinalizeContext())
            self._state.event_bus.trigger(EVENT_TRAIN_FINISHED, EventTrainFinishedContext())

    def sleep(self, tags: Iterable[SleepTag] = DEFAULT_SLEEP_TAGS) -> None:
        """Releases the GPU-resident training state selected by "tags" to host memory.

        This frees the GPU for a colocated workload, such as a rollout engine in colocated RL.
        The call is collective: every rank must invoke it with identical tags. Requesting a tag
        whose subsystem is already offloaded is a no-op.

        Args:
            tags: The subsystems to offload. Defaults to "SleepTag.TENSOR_STATES".

        Raises:
            NotImplementedError: If "SleepTag.COMMS" is requested, since it is not yet implemented.
            RuntimeError: If called during an in-flight gradient accumulation.
        """
        self._sleeper.sleep(tags)

    def wake(self, tags: Iterable[SleepTag] = DEFAULT_SLEEP_TAGS) -> None:
        """Restores GPU residency of the training state previously released by "sleep".

        The call is collective: every rank must invoke it with identical tags. Requesting a tag
        whose subsystem is not offloaded is a no-op.

        Args:
            tags: The subsystems to restore. Defaults to "SleepTag.TENSOR_STATES".

        Raises:
            NotImplementedError: If "SleepTag.COMMS" is requested, since it is not yet implemented.
        """
        self._sleeper.wake(tags)

    def is_sleeping(self, tag: SleepTag) -> bool:
        """Reports whether the subsystem identified by "tag" is currently offloaded.

        Args:
            tag: The subsystem to query.

        Returns:
            True if the subsystem is offloaded to host memory, False otherwise.
        """
        return self._sleeper.is_sleeping(tag)

    def export(self, export_to: Path, load_checkpoint: bool):
        """Exports the current model state to the specified directory.

        This handles the distributed saving logic, allowing the model to be
        reconstituted later or used for inference.

        Args:
            export_to: The directory path where the model artifacts will be saved.
            load_checkpoint: If True, attempts to load the latest checkpoint
                into the model before exporting.
        """
        if load_checkpoint:
            self._state.checkpointer.load_last_checkpoint(self._state)

        self._state.exporter.export(export_to)
