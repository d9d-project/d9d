from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from d9d.loop.component import ModelStageFactory, TrackedModules
from d9d.loop.config import ModelStageFactoryConfig, PipeliningConfig
from d9d.pipelining.api import PipelineStageInfo
from torch import nn


class _Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 4, bias=False)
        self.second = nn.Linear(4, 4, bias=False)
        self.second.weight = self.first.weight
        self.first.alias = self.first.weight

    def reset_parameters(self):
        # The tie must already hold while the provider initializes its model.
        assert self.first.weight is self.second.weight
        assert self.first.weight is self.first.alias
        nn.init.ones_(self.first.weight)


@pytest.mark.local
def test_materialization_preserves_tied_parameters():
    provider = Mock()
    provider.initialize_model_stage.side_effect = lambda _: SimpleNamespace(model=_Tied())
    context = SimpleNamespace(
        mesh_params=SimpleNamespace(is_distributed=False),
        current_device=torch.device("cpu"),
    )
    factory = ModelStageFactory(
        model_provider=provider,
        dist_context=context,
        config_model=ModelStageFactoryConfig(),
        config_pipelining=PipeliningConfig(),
        parameter_runtime=Mock(),
    )

    model = factory._build_model_stage(PipelineStageInfo(current_stage=0, num_stages=1))

    assert model.first.weight is model.second.weight
    assert model.first.weight is model.first.alias
    assert len(list(model.parameters())) == 1
    assert torch.equal(model.first.weight, torch.ones(4, 4))


@pytest.mark.local
def test_checkpoint_load_invalidates_runtime_weights(monkeypatch):
    module = nn.Linear(4, 4, bias=False)
    runtime = Mock()
    tracked = TrackedModules(
        dist_context=SimpleNamespace(mesh_params=SimpleNamespace(is_distributed=False)),
        modules=[module],
        stateful_predicate=lambda _name, _value: True,
        parameter_runtime=runtime,
    )
    original_load = module.load_state_dict

    def load(state_dict, **kwargs):
        runtime.invalidate_weights.assert_called_once_with()
        return original_load(state_dict, **kwargs)

    monkeypatch.setattr(module, "load_state_dict", load)

    tracked.load_state_dict({"pp_0_stage_0": {"weight": torch.ones(4, 4)}})

    assert torch.equal(module.weight, torch.ones(4, 4))


@pytest.mark.local
def test_parallelization_must_keep_tied_parameters():
    def _split_tie(context):
        # A separate parallelize_*() call replaces the parameter in one module only.
        context.model.second.weight = nn.Parameter(context.model.second.weight.detach().clone())

    provider = Mock()
    provider.initialize_model_stage.side_effect = lambda _: SimpleNamespace(model=_Tied())
    provider.parallelize_model_stage.side_effect = _split_tie
    context = SimpleNamespace(
        mesh_params=SimpleNamespace(is_distributed=True),
        current_device=torch.device("cpu"),
    )
    factory = ModelStageFactory(
        model_provider=provider,
        dist_context=context,
        config_model=ModelStageFactoryConfig(),
        config_pipelining=PipeliningConfig(),
        parameter_runtime=Mock(),
    )

    with pytest.raises(ValueError, match="split the tied parameter"):
        factory._build_model_stage(PipelineStageInfo(current_stage=0, num_stages=1))
