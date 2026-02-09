from pathlib import Path

from tqdm import tqdm

from d9d.core.dist_context import DeviceMeshParameters
from d9d.internals.determinism import set_seeds
from d9d.internals.pipeline_state import PipelineStateHandler
from d9d.loop.component import (
    BatchMaths,
    DataLoaderFactory,
    GradientClipper,
    GradientManager,
    JobLogger,
    JobProfiler,
    LossComputer,
    ManualGarbageCollector,
    ModelStageExporter,
    ModelStageFactory,
    OptimizerFactory,
    StateCheckpointer,
    Stepper,
    TimeoutManager,
    TrainTaskOperator,
)
from d9d.loop.config import TrainerConfig
from d9d.loop.control import (
    CreateMetricsContext,
    DatasetProvider,
    FinalizeContext,
    LRSchedulerProvider,
    ModelProvider,
    OptimizerProvider,
    TrainTaskProvider,
    TrainTaskProviderContext,
)
from d9d.loop.state import TrainJobState
from d9d.metric.impl import ComposeMetric


class TrainingConfigurator:
    """
    Orchestrates the assembly of the distributed training environment.

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
            data_provider: DatasetProvider,
            optimizer_provider: OptimizerProvider,
            lr_scheduler_provider: LRSchedulerProvider
    ):
        """
        Constructs a configurator capable of building the full training state.

        Args:
            mesh: Definition of the distributed device mesh topology.
            parameters: The global configuration object for the trainer.
            task_provider: Factory for creating the training task logic.
            model_provider: Factory for defining and creating model stages.
            data_provider: Factory for providing training datasets.
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

        timeout_manager = TimeoutManager(
            dist_context=dist_context,
            config=self._parameters.timeout
        )
        timeout_manager.set_init()

        task = self._task_provider(TrainTaskProviderContext(
            dist_context=dist_context
        ))

        batch_maths = BatchMaths(
            dist_context=dist_context,
            config_batching=self._parameters.batching,
            config_pipelining=self._parameters.pipelining
        )

        data_loader_factory = DataLoaderFactory(
            dist_context=dist_context,
            provider=self._data_provider,
            config_data_loading=self._parameters.data_loading,
            batch_maths=batch_maths
        )
        data_loader_train = data_loader_factory.build_dataloader_for_train_job()

        stepper = Stepper(
            initial_step=1,
            total_steps=len(data_loader_train)
        )

        pipeline_state_handler = PipelineStateHandler(
            sharding_spec={},
            num_shards=batch_maths.num_microbatches_pipelining
        )

        loss_computer = LossComputer(
            state=pipeline_state_handler,
            task=task,
            stepper=stepper
        )

        schedule, modules = ModelStageFactory(
            model_provider=self._model_provider,
            dist_context=dist_context,
            config_model=self._parameters.model_stage_factory,
            config_pipelining=self._parameters.pipelining,
            batch_maths=batch_maths,
            pipeline_callback=loss_computer
        ).build_pipeline_and_modules()

        metrics = ComposeMetric(task.create_metrics(CreateMetricsContext()).metrics)

        task_operator = TrainTaskOperator(
            dist_context=dist_context,
            task=task,
            pipeline=schedule,
            pipeline_state=pipeline_state_handler,
            metrics=metrics
        )

        grad_clipper = GradientClipper(
            dist_context=dist_context,
            tracked_modules=modules,
            config=self._parameters.gradient_clipping,
            stepper=stepper
        )

        optimizer, scheduler = OptimizerFactory(
            dist_context=dist_context,
            tracked_modules=modules,
            optimizer_provider=self._optimizer_provider,
            stepper=stepper,
            lr_scheduler_provider=self._lr_scheduler_provider
        ).build_optimizer_and_scheduler()

        gc = ManualGarbageCollector(
            dist_ctx=dist_context,
            config=self._parameters.gc,
            step=stepper
        )

        checkpointer = StateCheckpointer(
            dist_context=dist_context,
            stepper=stepper,
            config=self._parameters.checkpointing,
            gc=gc,
            run_name=self._parameters.run.name
        )

        profiler = JobProfiler(
            dist_context=dist_context,
            stepper=stepper,
            config=self._parameters.profiling
        )

        exporter = ModelStageExporter(
            model_provider=self._model_provider,
            dist_context=dist_context,
            modules=modules
        )

        gradient_manager = GradientManager(
            dist_context=dist_context,
            tracked_modules=modules,
            batch_maths=batch_maths,
            config=self._parameters.gradient_manager
        )

        job_logger = JobLogger(
            dist_context=dist_context,
            config=self._parameters.logging,
            metrics=metrics,
            stepper=stepper,
            run_config=self._parameters.run,
            additional_hparams={
                "task": task.dump_hparams(),
                "model": self._model_provider.dump_hparams()
            }
        )

        return TrainJobState(
            dist_context=dist_context,
            data_loader=data_loader_train,
            stepper=stepper,
            tracked_modules=modules,
            garbage_collector=gc,
            batch_maths=batch_maths,
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
            task_operator=task_operator
        )

    def configure(self) -> "Trainer":
        """
        Instantiates all training components and returns a configured Trainer.

        This method triggers the creation of the distributed context, sets seeds,
        builds the model, optimizer, data loaders, and attaches all auxiliary
        components (logging, profiling, checkpointing).

        Returns:
            Trainer: A ready-to-use trainer instance encapsulating the job state.
        """
        state = self._build_new_training_state()

        return Trainer(state)


class Trainer:
    """
    The main execution engine for running a distributed training job.

    This class manages the training loop, lifecycle events, distributed synchronization,
    and periodic side-effects (logging, checkpointing).
    """

    def __init__(self, state: TrainJobState):
        """
        Constructs a Trainer from a pre-built job state.

        Args:
            state: The encapsulated state object containing all initialized
                components (model, optimizer, dist_context, etc.).
        """
        self._state = state

    def train(self):
        """
        Executes the full training workflow.
        """
        self._state.dist_context.logger.info("Waiting for the world to start training")
        self._state.dist_context.wait_world()
        self._state.dist_context.logger.info("Trying to load last checkpoint before doing anything else")
        self._state.checkpointer.load_last_checkpoint(self._state)

        if self._state.stepper.current_step >= self._state.stepper.total_steps:
            self._state.dist_context.logger.info("Already trained fully, will do nothing")
            return

        self._state.dist_context.wait_world()

        with (
            tqdm(
                desc="Training",
                total=self._state.stepper.total_steps,
                disable=not self._state.dist_context.is_local_main_process,
                initial=self._state.stepper.current_step
            ) as bar,
            self._state.logger.new_run() as run,
            self._state.garbage_collector as gc,
            self._state.profiler.open() as profiler,
            self._state.gradient_manager.install(),
            self._state.gradient_clipper.install(),
            self._state.logger.install()
        ):
            run.set_context({"stage": "train"})

            for batch_group in self._state.data_loader:
                run.set_step(self._state.stepper.current_step)

                for batch in batch_group:
                    # we do both forward and backward passes
                    # since GradientManager is installed - it should start performing
                    # synchronization overlapping grad sync with compute
                    loss = self._state.task_operator.forward_backward(batch)

                    # add loss for grad manager - it want it for grad reduction
                    if loss is not None:
                        self._state.gradient_manager.add_loss_with_weight(loss.loss, loss.loss_weight)

                # metrics were successfully accumulated during forward passes - we can schedule their synchronization
                self._state.logger.trigger_sync()

                # wait for gradient synchronization finishes and scale them
                self._state.gradient_manager.sync_and_scale()

                # clip grads after they are synced across world
                self._state.gradient_clipper.clip_and_log(run)

                # optimize (it won't sync grads - they are already Replicate-d)
                self._state.optimizer.step()

                # update LR
                self._state.lr_scheduler.step()

                # log everything
                self._state.logger.log(
                    run,
                    loss_value=self._state.gradient_manager.compute_global_loss()
                )

                # reset grads
                self._state.gradient_manager.zero_grad()

                gc.collect_periodic()
                self._state.stepper.step()
                bar.update()

                # checkpoint at the end of the step
                self._state.checkpointer.checkpoint_if_needed(self._state)

                if profiler:
                    profiler.step()

                self._state.timeout_manager.set_periodic()

            self._state.task.finalize(FinalizeContext())

    def export(self, export_to: Path, load_checkpoint: bool):
        """
        Exports the current model state to the specified directory.

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
