import itertools
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.distributed.checkpoint.stateful import Stateful

from d9d.core.dist_context import REGULAR_DOMAIN, DistributedContext
from d9d.loop.config import ModelStageFactoryConfig, PipeliningConfig
from d9d.loop.control import InitializeModelStageContext, ModelProvider, ParallelizeModelStageContext
from d9d.model_state.io import load_model_state
from d9d.module.base import ModuleLateInit
from d9d.pipelining.api import PipelineStageInfo
from d9d.pipelining.factory.factory import PipelineScheduleInfo, build_schedule

from .batch_maths import BatchMaths
from .loss_computer import LossComputer

StatefulPredicate = Callable[[str, torch.Tensor], bool]
"""Determines if a specific parameter or buffer should be included in the state dictionary."""


def _stateful_predicate_requires_grad(key: str, value: torch.Tensor) -> bool:
    """Predicate that allows saving only tensors that require gradients."""
    return value.requires_grad


def _stateful_predicate_always(key: str, value: torch.Tensor) -> bool:
    """Predicate that always allows saving."""
    return True


class TrackedModules(Stateful):
    """
    Wraps a list of model stages and manages their state for distributed checkpointing.

    This class implements the PyTorch Distributed `Stateful` protocol, aggregating
    the state dictionaries of multiple pipeline stages assigned to the current rank.
    It handles namespacing to ensure uniqueness across pipeline ranks and stages.
    """

    def __init__(
            self,
            dist_context: DistributedContext,
            modules: list[nn.Module],
            stateful_predicate: StatefulPredicate
    ):
        """Constructs a TrackedModules object."""
        self._dist_context = dist_context
        self._modules = modules
        self._stateful_predicate = stateful_predicate

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forwards execution to the only pipeline stage.

        This method is only valid when pipeline parallelism is disabled.

        Args:
            *args: Positional arguments passed to the module.
            **kwargs: Keyword arguments passed to the module.

        Returns:
            The output of the model execution.

        Raises:
            ValueError: If pipeline parallelism is configured.
        """

        if self._dist_context.mesh_params.has_pipeline_parallel:
            raise ValueError("You cannot call tracked modules when using pipelining")

        return self._modules[0](*args, **kwargs)

    @property
    def modules(self) -> list[nn.Module]:
        """Returns the list of underlying PyTorch model modules."""
        return self._modules

    def _whitelisted_params(self, module: nn.Module) -> set[str]:
        allow_saving = set()
        for param_name, param in itertools.chain(module.named_parameters(), module.named_buffers()):
            if self._stateful_predicate(param_name, param):
                allow_saving.add(param_name)
        return allow_saving

    def _state_dict_stage(self, module: nn.Module) -> dict[str, Any]:
        whitelist = self._whitelisted_params(module)
        result = {
            k: v for k, v in module.state_dict().items() if k in whitelist
        }
        return result

    def state_dict(self) -> dict[str, Any]:
        """
        Generates the state dictionary for all tracked modules.

        The keys are namespaced using the current pipeline rank and stage index
        (e.g., `pp_0_stage_0`). Only parameters satisfying the `stateful_predicate`
        are included.

        Returns:
            A dictionary containing the states of all managed modules.
        """

        pp_rank = self._dist_context.mesh_for(REGULAR_DOMAIN)["pp"].get_local_rank()
        ret = {
            f"pp_{pp_rank}_stage_{i}": self._state_dict_stage(module)
            for i, module in enumerate(self._modules)
        }
        return ret

    def _load_state_dict_stage(self, module: nn.Module, state_dict: dict[str, Any]):
        whitelist = self._whitelisted_params(module)

        loading_result = module.load_state_dict(state_dict, strict=False)
        missing_keys = set(loading_result.missing_keys)
        extra_keys = set(loading_result.unexpected_keys)

        if len(whitelist.intersection(missing_keys)) > 0:
            raise ValueError(f"Missing keys: {whitelist.intersection(missing_keys)}")
        if len(extra_keys) > 0:
            raise ValueError(f"Extra keys: {extra_keys}")

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Loads the state dictionary into the tracked modules.

        Args:
            state_dict: The state dictionary to load. Must contain keys corresponding
                to the pipeline rank and stage indices managed by this instance.

        Raises:
            ValueError: If required keys are missing or unexpected keys are present
                based on the allow-list predicate.
        """

        pp_rank = self._dist_context.mesh_for(REGULAR_DOMAIN)["pp"].get_local_rank()
        for i, module in enumerate(self._modules):
            self._load_state_dict_stage(module, state_dict[f"pp_{pp_rank}_stage_{i}"])


class ModelStageFactory:
    """
    Factory class responsible for creating, initializing, and parallelizing model stages.

    This class coordinates the `ModelProvider` with the distributed context to:

    1. Initialize models on a meta device.
    2. Apply horizontal distribution strategy (TP, DP, FSDP, etc).
    3. Materialize weights on the target device.
    4. Load initial model states from checkpoints.
    """

    def __init__(
            self,
            model_provider: ModelProvider,
            dist_context: DistributedContext,
            batch_maths: BatchMaths,
            config_model: ModelStageFactoryConfig,
            config_pipelining: PipeliningConfig | None,
            loss_computer: LossComputer | None
    ):
        """Constructs a ModelStageFactory object."""

        self._model_provider = model_provider
        self._dist_context = dist_context
        self._config_model = config_model
        self._config_pipelining = config_pipelining
        self._batch_maths = batch_maths
        self._loss_computer = loss_computer

    def _build_model_stage(self, stage: PipelineStageInfo) -> nn.Module:
        # create a model with no real memory occupied
        with torch.device("meta"):
            factored = self._model_provider.initialize_model_stage(
                InitializeModelStageContext(
                    dist_context=self._dist_context,
                    stage=stage,
                )
            )

        model = factored.model

        if not isinstance(model, ModuleLateInit) or not isinstance(model, nn.Module):
            raise ValueError("Model stage is required to be nn.Module instance implementing ModuleLateInit protocol")

        # if current context is distributed - parallelize this model
        if self._dist_context.mesh_params.is_distributed:
            self._model_provider.parallelize_model_stage(
                ParallelizeModelStageContext(
                    model=model,
                    stage=stage,
                    dist_context=self._dist_context
                )
            )

        # move state that is bound to current device to it
        model.to_empty(device=self._dist_context.current_device)

        # reinitialize model parameters (only these are on current device)
        with torch.no_grad():
            model.reset_parameters()

        if self._config_model.source_checkpoint:
            load_model_state(
                src_dir=self._config_model.source_checkpoint,
                model=model,
                mapper=factored.state_mapper,
                device=f"cuda:{torch.cuda.current_device()}"
            )

        # set training state
        model.train()

        return model

    def build_pipeline_and_modules(
            self
    ) -> tuple[PipelineScheduleInfo | None, TrackedModules]:
        """
        Constructs the execution schedule and the model container.

        If pipeline parallelism is enabled, this orchestrates the creation of a
        distributed pipeline schedule.

        Otherwise, it simply builds a standalone model stage.

        Returns:
           The pipeline schedule information (or None if no pipelining).
           The `TrackedModules` instance wrapping the created model stage(s).

        Raises:
           ValueError: If pipelining configuration is missing but a pipeline is requested.
        """

        if self._config_model.checkpoint_only_trainable_parameters:
            stateful_predicate = _stateful_predicate_requires_grad
        else:
            stateful_predicate = _stateful_predicate_always

        if self._dist_context.mesh_params.has_pipeline_parallel:
            if self._config_pipelining is None:
                raise ValueError("Pipelining is enabled, but not configured")

            loss_fn = self._loss_computer.compute_loss_mul_weight if self._loss_computer is not None else None

            schedule, modules = build_schedule(
                dist_context=self._dist_context,
                n_microbatches=self._batch_maths.num_microbatches_pipelining,
                schedule_config=self._config_pipelining.schedule,
                model_provider=self._build_model_stage,
                loss_fn=loss_fn
            )

            return schedule, TrackedModules(self._dist_context, modules, stateful_predicate)
        else:
            model = self._build_model_stage(PipelineStageInfo(num_stages=1, current_stage=0))

            return None, TrackedModules(self._dist_context, [model], stateful_predicate)
