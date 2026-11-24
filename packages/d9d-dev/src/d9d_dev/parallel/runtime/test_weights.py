import pytest
import torch
from d9d.pipelining.api import mark_parameter_group, marked_parameter_groups
from torch import nn

from d9d_dev.parallel.runtime.helpers import MESHES, make_runtime, place_hsdp, run_one_region


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("trigger", ["rebind", "invalidate_weights", "zero_grad", "wait"])
def test_groups_read_new_sharded_weights(dist_ctx_factory, mesh, training, trigger):
    dist_ctx = dist_ctx_factory(mesh)
    model = nn.Linear(16, 17, bias=False, device="cuda")
    place_hsdp(model, dist_ctx)
    runtime = make_runtime(training=training)
    runtime.register(model, [])
    inputs = torch.ones((1, 16), device="cuda")

    def _forward():
        with torch.no_grad(), bound.parameters(model).group_weights(model, None):
            return model(inputs)

    bound = runtime.bind()
    try:
        _forward()
        # A checkpoint load writes the weights after bind() or invalidate_weights(). The optimizer writes them after
        # wait(), and an evaluation can run before zero_grad().
        match trigger:
            case "rebind":
                bound.close()
                bound = runtime.bind()
                model.weight.data.fill_(2)
            case "invalidate_weights":
                bound.invalidate_weights()
                model.weight.data.fill_(2)
            case "zero_grad":
                model.weight.data.fill_(2)
                bound.zero_grad()
            case "wait":
                run_one_region(bound, model, lambda: model(inputs).sum().backward() if training else None)
                model.weight.data.fill_(2)
        actual = _forward()
        torch.testing.assert_close(actual, torch.full_like(actual, 32))
    finally:
        bound.close()


class _TiedGroups(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = mark_parameter_group(nn.Linear(4, 4, bias=False, device="cuda"))
        self.second = mark_parameter_group(nn.Linear(4, 4, bias=False, device="cuda"))
        self.second.weight = self.first.weight

    async def forward_steps(self, steps, inputs):
        hidden = await steps.call(self.first, inputs)
        return await steps.call(self.second, hidden)


class _AliasGroup(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, inputs):
        return inputs * self.scale


class _AliasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(4, 4, device="cuda"))
        # The group holds the parameter that the module owns, as a shared embedding of a head does.
        self.alias = mark_parameter_group(_AliasGroup(self.scale))

    async def forward_steps(self, steps, inputs):
        return await steps.call(self.alias, inputs)


def _aliases():
    module = nn.Module()
    module.first = nn.Parameter(torch.ones(4, 4, device="cuda"))
    module.second = module.first
    return module, [(module, module, "first", "second")]


def _tied_groups():
    model = _TiedGroups()
    return model, [(model.first, model.second, "weight", "weight"), (model.second, model.first, "weight", "weight")]


def _alias_group():
    model = _AliasModel()
    return model, [(model.alias, model, "scale", "scale")]


@pytest.mark.local
@pytest.mark.parametrize("build", [_aliases, _tied_groups, _alias_group], ids=["aliases", "tied_groups", "alias_group"])
def test_one_parameter_has_one_unsharded_parameter(build):
    model, uses = build()
    originals = {id(param): param for param in model.parameters()}
    runtime = make_runtime()
    runtime.register(model, marked_parameter_groups(model))
    bound = runtime.bind()
    try:
        # The optimizer writes the sharded parameter, and every group that uses it must see the write.
        for param in originals.values():
            param.data.fill_(2)
        parameters = bound.parameters(model)
        for group, other, name, other_name in uses:
            with parameters.group_weights(group, None):
                param = getattr(group, name)
                assert param is getattr(other, other_name)
                assert torch.equal(param, torch.full_like(param, 2))
        assert {id(param): param for param in model.parameters()} == originals
    finally:
        bound.close()


@pytest.mark.local
def test_close_frees_what_bind_allocated():
    model = nn.Linear(1024, 1024, device="cuda")
    runtime = make_runtime()
    runtime.register(model, [])
    allocated = torch.cuda.memory_allocated()
    for _ in range(3):
        bound = runtime.bind()
        bound.zero_grad()
        run_one_region(bound, model, lambda: model(torch.ones(2, 1024, device="cuda")).sum().backward())
        bound.close()
        assert torch.cuda.memory_allocated() == allocated
