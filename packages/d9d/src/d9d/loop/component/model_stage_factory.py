import dataclasses

import torch
from torch import nn

from d9d.dist_context import BASE_DOMAIN, BaseDim, DistributedContext
from d9d.model_state.io import load_model_state
from d9d.nn import ModuleLateInit
from d9d.parallel.runtime import ParameterRuntime
from d9d.pipelining.api import PipelineStageInfo
from d9d.pipelining.factory import PipelineScheduleInfo, build_program_builder, build_schedule
from d9d.rng import fork_rng_child

from ..config import ModelStageFactoryConfig, PipeliningConfig
from ..control import InitializeModelStageContext, ModelProvider, ParallelizeModelStageContext
from .tracked_modules import StatefulPredicate, TrackedModules


def _stateful_predicate_requires_grad(key: str, value: torch.Tensor) -> bool:
    return value.requires_grad


def _stateful_predicate_always(key: str, value: torch.Tensor) -> bool:
    return True


@dataclasses.dataclass(frozen=True, slots=True)
class _TensorSlot:
    """A place that holds a parameter or a buffer: a module and the name of the tensor in it.

    Attributes:
        qualified_name: The name of the tensor in the model.
        module: The module that holds the tensor.
        name: The name of the tensor in the module.
        is_buffer: Whether the tensor is a buffer rather than a parameter.
    """

    qualified_name: str
    module: nn.Module
    name: str
    is_buffer: bool

    def get(self) -> torch.Tensor:
        """Returns the tensor that the slot holds now."""
        return self.module.get_buffer(self.name) if self.is_buffer else self.module.get_parameter(self.name)


def _tied_slots(model: nn.Module) -> list[list[_TensorSlot]]:
    """Returns the slots of each parameter or buffer held in several places, one list per tensor."""
    slots: dict[int, list[_TensorSlot]] = {}
    for prefix, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False, remove_duplicate=False):
            qualified_name = f"{prefix}.{name}" if prefix else name
            slots.setdefault(id(parameter), []).append(_TensorSlot(qualified_name, module, name, is_buffer=False))
        for name, buffer in module.named_buffers(recurse=False, remove_duplicate=False):
            qualified_name = f"{prefix}.{name}" if prefix else name
            slots.setdefault(id(buffer), []).append(_TensorSlot(qualified_name, module, name, is_buffer=True))
    return [tensor_slots for tensor_slots in slots.values() if len(tensor_slots) > 1]


def _to_empty_keeping_ties(model: nn.Module, device: torch.device):
    """Moves the model to a device with uninitialized storage, like ``nn.Module.to_empty()``, and keeps its ties.

    Args:
        model: The model to move.
        device: The device to allocate the parameters and buffers on.
    """
    # to_empty() replaces the tensors of each module on its own, so it unties a tensor that several modules hold.
    ties = _tied_slots(model)
    model.to_empty(device=device)
    for first, *others in ties:
        shared = first.get()
        for slot in others:
            # nn.Module.__setattr__() replaces a registered parameter or buffer and keeps the persistence of a buffer.
            setattr(slot.module, slot.name, shared)


class ModelStageFactory:
    """Factory that creates, initializes and parallelizes model stages.

    It drives the ``ModelProvider`` to:

    1.  Initialize models on the meta device.
    2.  Apply horizontal parallelism (TP, DP, FSDP, etc.).
    3.  Materialize weights on the target device.
    4.  Load the initial model state from a checkpoint.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        dist_context: DistributedContext,
        config_model: ModelStageFactoryConfig,
        config_pipelining: PipeliningConfig,
        parameter_runtime: ParameterRuntime,
    ):
        """Constructs the ``ModelStageFactory`` object.

        Args:
            model_provider: The provider that builds and parallelizes each model stage.
            dist_context: The distributed context.
            config_model: The model initialization configuration.
            config_pipelining: The pipeline parallelism configuration.
            parameter_runtime: The runtime that runs the stages.
        """
        self._model_provider = model_provider
        self._dist_context = dist_context
        self._config_model = config_model
        self._config_pipelining = config_pipelining
        self._runtime = parameter_runtime

    def _build_model_stage(self, stage: PipelineStageInfo) -> nn.Module:
        stage_index = stage.current_stage

        # Build on the meta device, so no memory is allocated yet.
        with fork_rng_child("initialize_model_stage", stage_index), torch.device("meta"):
            factored = self._model_provider.initialize_model_stage(
                InitializeModelStageContext(
                    dist_context=self._dist_context,
                    stage=stage,
                )
            )

        model = factored.model

        if not isinstance(model, ModuleLateInit) or not isinstance(model, nn.Module):
            raise ValueError(
                f"Model stage ({type(model).__name__}) must be an nn.Module that implements ModuleLateInit."
            )

        ties = _tied_slots(model)
        if self._dist_context.is_distributed:
            with fork_rng_child("parallelize_model_stage", stage_index):
                self._model_provider.parallelize_model_stage(
                    ParallelizeModelStageContext(model=model, stage=stage, dist_context=self._dist_context)
                )
            # distribute_parameters() keeps a tie only among the modules that one call gets.
            for slots in ties:
                if len({id(slot.get()) for slot in slots}) > 1:
                    raise ValueError(
                        f"The parallelization split the tied tensor ({[slot.qualified_name for slot in slots]}). "
                        "Distribute all modules that share it with one parallelize_*() call."
                    )

        _to_empty_keeping_ties(model, self._dist_context.current_device)

        # to_empty allocates uninitialized storage, so initialize the parameters.
        with torch.no_grad(), fork_rng_child("reset_parameters", stage_index):
            model.reset_parameters()

        if self._config_model.source_checkpoint:
            load_model_state(
                src_dir=self._config_model.source_checkpoint,
                model=model,
                mapper=factored.state_mapper,
                device=str(self._dist_context.current_device),
                position=self._dist_context.local_rank,  # Progress bar is displayed once per rank inside this node
            )

        model.train()

        return model

    def build_pipeline_and_modules(self) -> tuple[PipelineScheduleInfo, TrackedModules]:
        """Builds the pipeline schedule and the model stages.

        Returns:
            A tuple of the pipeline schedule info and the ``TrackedModules`` that wrap the model stages.
        """
        if self._config_model.checkpoint_only_trainable_parameters:
            stateful_predicate: StatefulPredicate = _stateful_predicate_requires_grad
        else:
            stateful_predicate = _stateful_predicate_always

        pp_mesh = self._dist_context.mesh_for(BASE_DOMAIN)[BaseDim.PP] if self._dist_context.is_distributed else None
        schedule, modules = build_schedule(
            pp_mesh=pp_mesh,
            program_builder=build_program_builder(self._config_pipelining.schedule),
            model_provider=self._build_model_stage,
            runtime=self._runtime,
        )

        return schedule, TrackedModules(
            self._dist_context, modules, stateful_predicate, parameter_runtime=self._runtime
        )
