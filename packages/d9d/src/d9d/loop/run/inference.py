import contextlib
from collections.abc import Iterator

import torch
from tqdm import tqdm

from d9d.data import MicrobatchPack, build_device_pack_stream
from d9d.dist_context import BATCH_DOMAIN, DeviceMeshParameters, DistributedContext
from d9d.pipelining.factory import PipelineScheduleInferenceConfig
from d9d.rng import SeedScope, fork_rng, fork_rng_child

from ..component import (
    DataParallelMicrobatchPackStream,
    InferenceTaskOperator,
    JobProfiler,
    JobSchedule,
    ManualGarbageCollector,
    ModelStageFactory,
    PipelineStateHandler,
    StateCheckpointer,
    TimeoutManager,
    build_parameter_runtime,
)
from ..config import InferenceConfig, PipeliningConfig
from ..control import (
    DataProvider,
    FinalizeContext,
    InferenceTaskProvider,
    InferenceTaskProviderContext,
    InitializeDataProviderContext,
    ModelProvider,
    RegisterModelEventsContext,
    RegisterTaskEventsContext,
)
from ..event import EventBus
from ..event.catalogue.common import (
    EventConfigurationStartedContext,
    EventDataStreamReadyContext,
    EventModelStagesReadyContext,
    EventStepContext,
)
from ..event.catalogue.inference import (
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
from ..state import InferenceJobState


class InferenceConfigurator:
    """Configurator that assembles the distributed inference environment.

    It combines the device mesh parameters, the ``InferenceConfig`` and the user-defined providers
    into an ``Inference`` object that is ready to run the inference loop.
    """

    def __init__(
        self,
        mesh: DeviceMeshParameters,
        parameters: InferenceConfig,
        task_provider: InferenceTaskProvider,
        model_provider: ModelProvider,
        data_provider: DataProvider,
    ):
        """Constructs the ``InferenceConfigurator`` object.

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

    def _build_new_state(self, dist_context: DistributedContext, seed_scope: SeedScope) -> InferenceJobState:
        pipelining_config = PipeliningConfig(schedule=PipelineScheduleInferenceConfig())

        timeout_manager = TimeoutManager(dist_context=dist_context, config=self._parameters.timeout)
        timeout_manager.set_init()

        with fork_rng_child("task_provider"):
            task = self._task_provider(InferenceTaskProviderContext(dist_context=dist_context))

        event_bus = EventBus()
        with fork_rng_child("register_events", "model_provider"):
            self._model_provider.register_events(
                RegisterModelEventsContext(dist_context=dist_context, event_bus=event_bus)
            )
        with fork_rng_child("register_events", "task"):
            task.register_events(RegisterTaskEventsContext(dist_context=dist_context, event_bus=event_bus))

        event_bus.trigger(EVENT_INFERENCE_CONFIG_STARTED, EventConfigurationStartedContext(dist_context=dist_context))

        dp_rank = (
            dist_context.mesh_for(BATCH_DOMAIN)["dp"].get_local_rank() if dist_context.mesh_params.is_distributed else 0
        )
        data_seed_scope = seed_scope.child("data")
        with fork_rng_child("data_provider"):
            microbatch_pack_stream = self._data_provider(
                InitializeDataProviderContext(
                    dist_context=dist_context,
                    seed_scope=data_seed_scope,
                    shard_seed_scope=data_seed_scope.child("dp", dp_rank),
                )
            )
        event_bus.trigger(EVENT_INFERENCE_DATA_STREAM_READY, EventDataStreamReadyContext(stream=microbatch_pack_stream))

        checkpointable_stream = DataParallelMicrobatchPackStream(
            dist_context=dist_context,
            inner=microbatch_pack_stream,
        )

        device_pack_stream = build_device_pack_stream(
            stream=checkpointable_stream,
            device=dist_context.current_device,
            prefetch_factor=self._parameters.data_prefetch.prefetch_factor,
        )

        schedule = JobSchedule(
            config=self._parameters.schedule,
            stream=checkpointable_stream,
        )

        pipeline_state_handler = PipelineStateHandler()

        parameter_runtime = build_parameter_runtime(self._parameters.parameter_runtime)
        pipeline_schedule, modules = ModelStageFactory(
            model_provider=self._model_provider,
            dist_context=dist_context,
            config_model=self._parameters.model_stage_factory,
            config_pipelining=pipelining_config,
            parameter_runtime=parameter_runtime,
        ).build_pipeline_and_modules()
        event_bus.trigger(EVENT_INFERENCE_MODEL_STAGES_READY, EventModelStagesReadyContext(modules=modules.modules))

        task_operator = InferenceTaskOperator(
            dist_context=dist_context,
            task=task,
            pipeline=pipeline_schedule,
            pipeline_state=pipeline_state_handler,
        )

        gc = ManualGarbageCollector(dist_context=dist_context, config=self._parameters.gc, schedule=schedule)

        checkpointer = StateCheckpointer(
            dist_context=dist_context, schedule=schedule, config=self._parameters.checkpointing, gc=gc, run_name=None
        )

        profiler = JobProfiler(dist_context=dist_context, schedule=schedule, config=self._parameters.profiling)

        return InferenceJobState(
            dist_context=dist_context,
            seed_scope=seed_scope,
            microbatch_pack_stream=device_pack_stream,
            schedule=schedule,
            tracked_modules=modules,
            garbage_collector=gc,
            checkpointer=checkpointer,
            task=task,
            profiler=profiler,
            timeout_manager=timeout_manager,
            task_operator=task_operator,
            event_bus=event_bus,
            parameter_runtime=parameter_runtime,
        )

    def configure(self) -> "Inference":
        """Builds all inference components and returns a configured ``Inference`` object.

        It creates the distributed context and builds the model, the data stream and the auxiliary components.

        Returns:
            A ready-to-use inference engine that holds the job state.
        """
        # Build the mesh outside the fork: it sets the current CUDA device, whose generator the fork saves.
        dist_context = self._mesh.build()
        seed_scope = SeedScope.root(self._parameters.determinism.base_seed)

        with fork_rng(seed_scope.child("configure")):
            state = self._build_new_state(dist_context, seed_scope)

        return Inference(state)


class Inference:
    """The main execution engine for running a distributed inference job.

    This class manages the inference loop, lifecycle events, distributed synchronization,
    and periodic side effects (profiling, checkpointing). The model runs in evaluation mode
    inside ``torch.inference_mode``.
    """

    def __init__(self, state: InferenceJobState):
        """Constructs the ``Inference`` object from a pre-built job state.

        Args:
            state: The encapsulated state object containing all initialized components.
        """
        self._state = state

    def _enable_eval_mode(self):
        for module in self._state.tracked_modules.modules.values():
            module.eval()

    def infer(self):
        """Executes the full inference workflow.

        This method:

        1.  Waits for all ranks.
        2.  Loads the latest checkpoint if available.
        3.  Iterates through the data stream.
        4.  Runs the pipeline forward pass for every pack.
        5.  Runs periodic garbage collection and profiling.
        6.  Finalizes the task on completion.

        Raises:
            RuntimeError: If the data stream ends before ``total_steps``.
        """
        with torch.inference_mode(), fork_rng(self._state.seed_scope.child("infer")):
            self._infer()

    def _infer(self):
        self._enable_eval_mode()

        self._state.dist_context.wait_world()
        self._state.dist_context.logger.info("Trying to load last checkpoint before doing anything else")
        self._state.checkpointer.load_last_checkpoint(self._state)

        if self._state.schedule.current_step >= self._state.schedule.total_steps:
            self._state.dist_context.logger.info("Inference is already complete, nothing to do")
            return

        self._state.dist_context.wait_world()

        step_ctx = EventStepContext(schedule=self._state.schedule)

        # Bound only around the loop, so that the model stages are final, as in training.
        self._state.parameter_runtime.bind(training=False)
        try:
            self._run_loop(step_ctx)
        finally:
            self._state.parameter_runtime.unbind()

    def _run_loop(self, step_ctx: EventStepContext):
        with (
            tqdm(
                desc="Inference",
                total=self._state.schedule.total_steps,
                disable=not self._state.dist_context.is_local_main_process,
                initial=self._state.schedule.current_step,
            ) as bar,
            self._state.garbage_collector,
            self._state.profiler.open() as profiler,
            contextlib.closing(iter(self._state.microbatch_pack_stream)) as packs,
        ):
            self._state.event_bus.trigger(EVENT_INFERENCE_READY, EventInferenceReadyContext())

            while self._state.schedule.current_step < self._state.schedule.total_steps:
                with fork_rng_child("step", self._state.schedule.current_step):
                    self._infer_step(packs, step_ctx)
                self._state.schedule.step()

                # Checkpoint after schedule.step(), so the saved step counter includes this step.
                self._state.checkpointer.checkpoint_if_needed(self._state)

                # End the profiled step only now, so that it covers the step post events and the checkpoint.
                if profiler:
                    profiler.step()

                bar.update()

            with fork_rng_child("finalize"):
                self._state.task.finalize(FinalizeContext())
            self._state.event_bus.trigger(EVENT_INFERENCE_FINISHED, EventInferenceFinishedContext())

    def _infer_step(self, packs: Iterator[MicrobatchPack], step_ctx: EventStepContext):
        self._state.event_bus.trigger(EVENT_INFERENCE_STEP_PRE, step_ctx)

        device_pack = next(packs, None)
        if device_pack is None:
            raise RuntimeError(
                f"The data stream ended at step ({self._state.schedule.current_step}) "
                f"before total_steps ({self._state.schedule.total_steps}). "
                "Lower total_steps or provide more data."
            )

        with self._state.event_bus.bounded(EVENT_INFERENCE_FORWARD_PRE, EVENT_INFERENCE_FORWARD_POST, step_ctx):
            self._state.task_operator.forward(device_pack)

        self._state.garbage_collector.collect_periodic()

        self._state.timeout_manager.set_periodic()

        self._state.event_bus.trigger(EVENT_INFERENCE_STEP_POST, step_ctx)
