import pytest
import torch
from d9d.dist_context import DenseDomainConfig, DeviceMeshParameters
from d9d.loop.component import ModelStageFactory, TrackedModules, build_parameter_runtime
from d9d.loop.config import ModelStageFactoryConfig, ParameterRuntimeConfig, PipeliningConfig
from d9d.loop.control import (
    InitializeModelStageContext,
    InitializeModelStageResult,
    ModelProvider,
    ParallelizeModelStageContext,
)
from d9d.model_state.mapper.adapters import identity_mapper_from_module
from d9d.rng import SeedScope, fork_rng
from torch import nn


class _Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 4, bias=False)
        self.second = nn.Linear(4, 4, bias=False)
        self.second.weight = self.first.weight
        self.first.alias = self.first.weight
        self.first.register_buffer("scale", torch.empty(4))
        self.second.register_buffer("scale", self.first.scale, persistent=False)

    def reset_parameters(self):
        # The ties must already hold while the model initializes its tensors.
        assert self.first.weight is self.second.weight
        assert self.first.weight is self.first.alias
        assert self.first.scale is self.second.scale


class _TiedProvider(ModelProvider[_Tied]):
    def initialize_model_stage(self, context: InitializeModelStageContext) -> InitializeModelStageResult[_Tied]:
        model = _Tied()
        return InitializeModelStageResult(model=model, state_mapper=identity_mapper_from_module(model))

    def parallelize_model_stage(self, context: ParallelizeModelStageContext[_Tied]):
        pass


@pytest.mark.local
def test_build_keeps_tied_parameters_and_buffers(dist_ctx_factory):
    factory = ModelStageFactory(
        model_provider=_TiedProvider(),
        dist_context=dist_ctx_factory(DeviceMeshParameters()),
        config_model=ModelStageFactoryConfig(),
        config_pipelining=PipeliningConfig(),
        parameter_runtime=build_parameter_runtime(ParameterRuntimeConfig()),
    )

    with fork_rng(SeedScope.root(0)):
        _, modules = factory.build_pipeline_and_modules()
    model = modules.modules[0]

    assert model.first.weight is model.second.weight
    assert model.first.weight is model.first.alias
    assert model.first.scale is model.second.scale
    state_dict = model.state_dict()
    assert "first.scale" in state_dict
    assert "second.scale" not in state_dict


class _SplittingProvider(_TiedProvider):
    def parallelize_model_stage(self, context: ParallelizeModelStageContext[_Tied]):
        # A separate parallelize_*() call replaces the parameter in one module only.
        context.model.second.weight = nn.Parameter(context.model.second.weight.detach().clone())


@pytest.mark.distributed
def test_parallelization_must_keep_tied_parameters(dist_ctx_factory):
    factory = ModelStageFactory(
        model_provider=_SplittingProvider(),
        dist_context=dist_ctx_factory(
            DeviceMeshParameters(data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=8))
        ),
        config_model=ModelStageFactoryConfig(),
        config_pipelining=PipeliningConfig(),
        parameter_runtime=build_parameter_runtime(ParameterRuntimeConfig()),
    )

    with pytest.raises(ValueError, match="split the tied tensor"), fork_rng(SeedScope.root(0)):
        factory.build_pipeline_and_modules()


@pytest.mark.local
def test_checkpoint_load_refreshes_runtime_weights(dist_ctx_factory):
    module = nn.Linear(4, 4, bias=False, device="cuda")
    runtime = build_parameter_runtime(ParameterRuntimeConfig(compute_dtype="bfloat16"))
    runtime.register_stage(0, module)
    tracked = TrackedModules(dist_ctx_factory(DeviceMeshParameters()), {0: module}, lambda _name, _value: True, runtime)
    runtime.bind()
    try:
        # The first region of a step makes the bf16 copy of the weight.
        with runtime.group_weights(0, 0):
            pass
        tracked.load_state_dict({"stage_0": {"weight": torch.full((4, 4), 2.0, device="cuda")}})
        with runtime.group_weights(0, 0):
            assert torch.equal(module.weight, torch.full_like(module.weight, 2))
    finally:
        runtime.unbind()
