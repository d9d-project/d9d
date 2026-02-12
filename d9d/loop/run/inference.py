import torch
from tqdm import tqdm

from d9d.core.dist_context import DeviceMeshParameters
from d9d.internals.determinism import set_seeds
from d9d.internals.pipeline_state import PipelineStateHandler
from d9d.loop.component import (
    BatchMaths,
    DataLoaderFactory,
    InferenceProcessor,
    InferenceTaskOperator,
    JobProfiler,
    ManualGarbageCollector,
    ModelStageFactory,
    StateCheckpointer,
    Stepper,
    TimeoutManager,
)
from d9d.loop.config import InferenceConfig, PipeliningConfig
from d9d.loop.control import (
    DatasetProvider,
    FinalizeContext,
    InferenceTaskProvider,
    InferenceTaskProviderContext,
    ModelProvider,
)
from d9d.loop.state import InferenceJobState
from d9d.pipelining.factory import PipelineScheduleInferenceConfig


class InferenceConfigurator:
    """
    Orchestrates the assembly of the distributed inference environment.

    This class binds the infrastructure configuration (DeviceMesh), the inference
    parameters, and the user-defined logic (Providers) to create a fully
    initialized state object capable of running the inference loop.
    """

    def __init__(
        self,
        mesh: DeviceMeshParameters,
        parameters: InferenceConfig,
        task_provider: InferenceTaskProvider,
        model_provider: ModelProvider,
        data_provider: DatasetProvider,
    ):
        """
        Constructs a configurator capable of building the full inference state.

        Args:
            mesh: Definition of the distributed device mesh topology.
            parameters: The global configuration object for inference.
            task_provider: Factory for creating the inference task logic.
            model_provider: Factory for defining and creating model stages.
            data_provider: Factory for providing inference datasets.
        """

        self._mesh = mesh
        self._parameters = parameters
        self._task_provider = task_provider
        self._model_provider = model_provider
        self._data_provider = data_provider

    def _build_new_state(self) -> InferenceJobState:
        dist_context = self._mesh.build()

        pipelining_config = PipeliningConfig(schedule=PipelineScheduleInferenceConfig())

        set_seeds(dist_context, seed=self._parameters.determinism.base_seed)

        timeout_manager = TimeoutManager(dist_context=dist_context, config=self._parameters.timeout)
        timeout_manager.set_init()

        task = self._task_provider(InferenceTaskProviderContext(dist_context=dist_context))

        batch_maths = BatchMaths(
            dist_context=dist_context, config_batching=self._parameters.batching, config_pipelining=pipelining_config
        )

        data_loader_factory = DataLoaderFactory(
            dist_context=dist_context,
            provider=self._data_provider,
            config_data_loading=self._parameters.data_loading,
            batch_maths=batch_maths,
        )
        data_loader_infer = data_loader_factory.build_dataloader_for_infer_job()

        stepper = Stepper(initial_step=1, total_steps=len(data_loader_infer))

        pipeline_state_handler = PipelineStateHandler(
            sharding_spec={}, num_shards=batch_maths.num_microbatches_pipelining
        )

        processor = InferenceProcessor(state=pipeline_state_handler, task=task)

        schedule, modules = ModelStageFactory(
            model_provider=self._model_provider,
            dist_context=dist_context,
            config_model=self._parameters.model_stage_factory,
            config_pipelining=pipelining_config,
            batch_maths=batch_maths,
            pipeline_callback=processor,
        ).build_pipeline_and_modules()

        task_operator = InferenceTaskOperator(
            dist_context=dist_context, task=task, pipeline=schedule, pipeline_state=pipeline_state_handler
        )

        gc = ManualGarbageCollector(dist_ctx=dist_context, config=self._parameters.gc, step=stepper)

        checkpointer = StateCheckpointer(
            dist_context=dist_context, stepper=stepper, config=self._parameters.checkpointing, gc=gc, run_name=None
        )

        profiler = JobProfiler(dist_context=dist_context, stepper=stepper, config=self._parameters.profiling)

        return InferenceJobState(
            dist_context=dist_context,
            data_loader=data_loader_infer,
            stepper=stepper,
            tracked_modules=modules,
            garbage_collector=gc,
            batch_maths=batch_maths,
            checkpointer=checkpointer,
            task=task,
            profiler=profiler,
            timeout_manager=timeout_manager,
            task_operator=task_operator,
        )

    def configure(self) -> "Inference":
        """
        Instantiates all inference components and returns a configured Inference engine.

        This method triggers the creation of the distributed context, sets seeds,
        builds the model, data loaders, and attaches all auxiliary components.

        Returns:
            Inference: A ready-to-use inference engine instance encapsulating the job state.
        """

        state = self._build_new_state()

        return Inference(state)


class Inference:
    """
    The main execution engine for running a distributed inference job.

    This class manages the inference loop, lifecycle events, distributed synchronization,
    and periodic side-effects (profiling, checkpointing). It ensures the model is in
    evaluation mode and runs within a `torch.inference_mode` context.
    """

    def __init__(self, state: InferenceJobState):
        """
        Constructs an Inference engine from a pre-built job state.

        Args:
            state: The encapsulated state object containing all initialized components.
        """

        self._state = state

    def _enable_eval_mode(self):
        for module in self._state.tracked_modules.modules:
            module.eval()

    def infer(self):
        """
        Executes the full inference workflow.

        This method:

        1. Waits for world synchronization.
        2. Loads the latest checkpoint if available.
        3. Iterates through the data loader.
        4. Executes the pipeline forward pass for every batch.
        5. Handles periodic garbage collection and profiling.
        6. Finalizes the task upon completion.
        """

        with torch.inference_mode():
            self._enable_eval_mode()

            self._state.dist_context.logger.info("Waiting for the world to start job")
            self._state.dist_context.wait_world()
            self._state.dist_context.logger.info("Trying to load last checkpoint before doing anything else")
            self._state.checkpointer.load_last_checkpoint(self._state)

            if self._state.stepper.current_step >= self._state.stepper.total_steps:
                self._state.dist_context.logger.info("Already ran, will do nothing")
                return

            self._state.dist_context.wait_world()

            with (
                tqdm(
                    desc="Inference",
                    total=self._state.stepper.total_steps,
                    disable=not self._state.dist_context.is_local_main_process,
                    initial=self._state.stepper.current_step,
                ) as bar,
                self._state.garbage_collector as gc,
                self._state.profiler.open() as profiler,
            ):
                for batch_group in self._state.data_loader:
                    for batch in batch_group:
                        self._state.task_operator.forward(batch)

                    gc.collect_periodic()
                    self._state.stepper.step()
                    bar.update()

                    # checkpoint at the end of the step
                    self._state.checkpointer.checkpoint_if_needed(self._state)

                    if profiler:
                        profiler.step()

                    self._state.timeout_manager.set_periodic()

                self._state.task.finalize(FinalizeContext())
