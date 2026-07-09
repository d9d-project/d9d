import torch
from tqdm import tqdm

from d9d.core.dist_context import DeviceMeshParameters
from d9d.internals.determinism import set_seeds
from d9d.loop.component import (
    DataParallelMicrobatchPackStream,
    InferenceTaskOperator,
    JobProfiler,
    JobSchedule,
    ManualGarbageCollector,
    ModelStageFactory,
    PipelineStateHandler,
    StateCheckpointer,
    TimeoutManager,
)
from d9d.loop.config import InferenceConfig, PipeliningConfig
from d9d.loop.control import (
    DataProvider,
    FinalizeContext,
    InferenceTaskProvider,
    InferenceTaskProviderContext,
    InitializeDataProviderContext,
    ModelProvider,
    RegisterModelEventsContext,
    RegisterTaskEventsContext,
)
from d9d.loop.event import (
    EventBus,
)
from d9d.loop.event.catalogue.common import (
    EventConfigurationStartedContext,
    EventDataStreamReadyContext,
    EventModelStagesReadyContext,
    EventStepContext,
)
from d9d.loop.event.catalogue.inference import (
    EVENT_INFERENCE_CONFIG_STARTED,
    EVENT_INFERENCE_DATA_STREAM_READY,
    EVENT_INFERENCE_FINISHED,
    EVENT_INFERENCE_FORWARD_POST,
    EVENT_INFERENCE_FORWARD_PRE,
    EVENT_INFERENCE_MODEL_STAGES_READY,
    EVENT_INFERENCE_READY,
    EVENT_INFERENCE_STEP_POST,
    EVENT_INFERENCE_STEP_PRE,
    EventInferenceFinishedContext,
    EventInferenceReadyContext,
)
from d9d.loop.state import InferenceJobState
from d9d.pipelining.factory import PipelineScheduleInferenceConfig

from ._device import move_pack_to_device


class InferenceConfigurator:
    """Orchestrates the assembly of the distributed inference environment.

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
        data_provider: DataProvider,
    ):
        """Constructs a configurator capable of building the full inference state.

        Args:
            mesh: Definition of the distributed device mesh topology.
            parameters: The global configuration object for inference.
            task_provider: Factory for creating the inference task logic.
            model_provider: Factory for defining and creating model stages.
            data_provider: Factory for building the microbatch pack stream.
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

        event_bus = EventBus()
        self._model_provider.register_events(RegisterModelEventsContext(dist_context=dist_context, event_bus=event_bus))
        task.register_events(RegisterTaskEventsContext(dist_context=dist_context, event_bus=event_bus))

        event_bus.trigger(EVENT_INFERENCE_CONFIG_STARTED, EventConfigurationStartedContext(dist_context=dist_context))

        microbatch_pack_stream = self._data_provider(InitializeDataProviderContext(dist_context=dist_context))
        event_bus.trigger(EVENT_INFERENCE_DATA_STREAM_READY, EventDataStreamReadyContext(stream=microbatch_pack_stream))

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
            config_pipelining=pipelining_config,
        ).build_pipeline_and_modules()
        event_bus.trigger(EVENT_INFERENCE_MODEL_STAGES_READY, EventModelStagesReadyContext(modules=modules.modules))

        task_operator = InferenceTaskOperator(
            dist_context=dist_context, task=task, pipeline=pipeline_schedule, pipeline_state=pipeline_state_handler
        )

        gc = ManualGarbageCollector(dist_ctx=dist_context, config=self._parameters.gc, schedule=schedule)

        checkpointer = StateCheckpointer(
            dist_context=dist_context, schedule=schedule, config=self._parameters.checkpointing, gc=gc, run_name=None
        )

        profiler = JobProfiler(dist_context=dist_context, schedule=schedule, config=self._parameters.profiling)

        return InferenceJobState(
            dist_context=dist_context,
            microbatch_pack_stream=checkpointable_stream,
            schedule=schedule,
            tracked_modules=modules,
            garbage_collector=gc,
            checkpointer=checkpointer,
            task=task,
            profiler=profiler,
            timeout_manager=timeout_manager,
            task_operator=task_operator,
            event_bus=event_bus,
        )

    def configure(self) -> "Inference":
        """Instantiates all inference components and returns a configured Inference engine.

        This method triggers the creation of the distributed context, sets seeds,
        builds the model, data loaders, and attaches all auxiliary components.

        Returns:
            Inference: A ready-to-use inference engine instance encapsulating the job state.
        """
        state = self._build_new_state()

        return Inference(state)


class Inference:
    """The main execution engine for running a distributed inference job.

    This class manages the inference loop, lifecycle events, distributed synchronization,
    and periodic side-effects (profiling, checkpointing). It ensures the model is in
    evaluation mode and runs within a `torch.inference_mode` context.
    """

    def __init__(self, state: InferenceJobState):
        """Constructs an Inference engine from a pre-built job state.

        Args:
            state: The encapsulated state object containing all initialized components.
        """
        self._state = state

    def _enable_eval_mode(self):
        for module in self._state.tracked_modules.modules:
            module.eval()

    def infer(self):
        """Executes the full inference workflow.

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

            self._state.dist_context.wait_world()
            self._state.dist_context.logger.info("Trying to load last checkpoint before doing anything else")
            self._state.checkpointer.load_last_checkpoint(self._state)

            if self._state.schedule.current_step >= self._state.schedule.total_steps:
                self._state.dist_context.logger.info("Already ran, will do nothing")
                return

            self._state.dist_context.wait_world()

            step_ctx = EventStepContext(schedule=self._state.schedule)

            with (
                tqdm(
                    desc="Inference",
                    total=self._state.schedule.total_steps,
                    disable=not self._state.dist_context.is_local_main_process,
                    initial=self._state.schedule.current_step,
                ) as bar,
                self._state.garbage_collector as gc,
                self._state.profiler.open() as profiler,
            ):
                self._state.event_bus.trigger(EVENT_INFERENCE_READY, EventInferenceReadyContext())

                for pack in self._state.microbatch_pack_stream:
                    self._state.event_bus.trigger(EVENT_INFERENCE_STEP_PRE, step_ctx)

                    device_pack = move_pack_to_device(pack, "cuda")

                    with self._state.event_bus.bounded(
                        EVENT_INFERENCE_FORWARD_PRE, EVENT_INFERENCE_FORWARD_POST, step_ctx
                    ):
                        self._state.task_operator.forward(device_pack)

                    gc.collect_periodic()

                    if profiler:
                        profiler.step()

                    self._state.timeout_manager.set_periodic()

                    self._state.event_bus.trigger(EVENT_INFERENCE_STEP_POST, step_ctx)
                    self._state.schedule.step()

                    # checkpoint at the end of the step
                    self._state.checkpointer.checkpoint_if_needed(self._state)

                    bar.update()

                self._state.task.finalize(FinalizeContext())
                self._state.event_bus.trigger(EVENT_INFERENCE_FINISHED, EventInferenceFinishedContext())
