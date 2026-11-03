import torch
from torch import nn

from d9d.dist_context import REGULAR_DOMAIN, DistributedContext
from d9d.model_state.io import load_model_state
from d9d.nn import ModuleLateInit
from d9d.parallel.runtime import ParameterRuntime
from d9d.pipelining.api import PipelineStageInfo
from d9d.pipelining.factory import PipelineScheduleInfo, build_program_builder, build_schedule

from ..config import ModelStageFactoryConfig, PipeliningConfig
from ..control import InitializeModelStageContext, ModelProvider, ParallelizeModelStageContext
from .tracked_modules import StatefulPredicate, TrackedModules


def _stateful_predicate_requires_grad(key: str, value: torch.Tensor) -> bool:
    return value.requires_grad


def _stateful_predicate_always(key: str, value: torch.Tensor) -> bool:
    return True


def _parameter_owners(model: nn.Module) -> list[list[tuple[str, nn.Module, str]]]:
    # Each entry lists the qualified name, the module and the local name of every place that holds one parameter.
    owners: dict[int, list[tuple[str, nn.Module, str]]] = {}
    for prefix, module in model.named_modules(remove_duplicate=False):
        for name, parameter in module.named_parameters(recurse=False, remove_duplicate=False):
            qualified = f"{prefix}.{name}" if prefix else name
            owners.setdefault(id(parameter), []).append((qualified, module, name))
    return list(owners.values())


def _check_ties_kept(ties: list[list[tuple[str, nn.Module, str]]]):
    for places in ties:
        _, first_module, first_name = places[0]
        first = first_module.get_parameter(first_name)
        if any(module.get_parameter(name) is not first for _, module, name in places[1:]):
            names = [qualified for qualified, _, _ in places]
            raise ValueError(
                f"The parallelization split the tied parameter ({names}) into separate parameters. "
                "Distribute the modules that share it with one parallelize_*() call."
            )


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
        # Build on the meta device, so no memory is allocated yet.
        with torch.device("meta"):
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

        if self._dist_context.mesh_params.is_distributed:
            ties = [places for places in _parameter_owners(model) if len(places) > 1]
            self._model_provider.parallelize_model_stage(
                ParallelizeModelStageContext(model=model, stage=stage, dist_context=self._dist_context)
            )
            # A parallelize_*() call replaces a parameter only in the modules it gets, so separate calls split a tie.
            _check_ties_kept(ties)

        # Moving ordinary meta parameters replaces them, so restore the ties before initialization.
        ties = [places for places in _parameter_owners(model) if len(places) > 1]
        model.to_empty(device=self._dist_context.current_device)
        for (_, owner, owner_name), *aliases in ties:
            for _, module, name in aliases:
                module.register_parameter(name, owner.get_parameter(owner_name))

        # to_empty allocates uninitialized storage, so initialize the parameters.
        with torch.no_grad():
            model.reset_parameters()

        if self._config_model.source_checkpoint:
            load_model_state(
                src_dir=self._config_model.source_checkpoint,
                model=model,
                mapper=factored.state_mapper,
                device=f"cuda:{torch.cuda.current_device()}",
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

        pp_mesh = (
            self._dist_context.mesh_for(REGULAR_DOMAIN)["pp"] if self._dist_context.mesh_params.is_distributed else None
        )
        schedule, modules = build_schedule(
            pp_mesh=pp_mesh,
            program_builder=build_program_builder(self._config_pipelining.schedule),
            model_provider=self._build_model_stage,
            runtime=self._runtime,
        )

        return schedule, TrackedModules(
            self._dist_context, modules, stateful_predicate, parameter_runtime=self._runtime
        )
