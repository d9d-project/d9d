import copy

import pytest
import torch
import torch.distributed as dist
from d9d.pipelining.api import mark_parameter_group, marked_parameter_groups
from torch import nn
from torch.distributed.tensor import DTensor

from d9d_dev.nn.helper.compare import assert_angle_and_norm_close
from d9d_dev.nn.helper.tolerances import grad_tolerance_for
from d9d_dev.parallel.runtime.helpers import MESHES, local_grad, make_runtime, place_hsdp

_HIDDEN = 17


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(_HIDDEN, _HIDDEN, device="cuda")
        self.unused = nn.Parameter(torch.ones(5, device="cuda"))


class _Stage(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(_HIDDEN, _HIDDEN, device="cuda") * 0.3)
        self.scale = nn.Parameter(torch.randn(_HIDDEN, device="cuda"))
        self.frozen = nn.Parameter(torch.randn(_HIDDEN, device="cuda"), requires_grad=False)
        self.first = mark_parameter_group(_Layer())
        self.second = mark_parameter_group(_Layer())
        self.second.linear.weight = self.first.linear.weight

    # Only marks the stage as resumable. The test drives the regions itself.
    async def forward_steps(self, steps, inputs):
        raise NotImplementedError

    def region_modules(self):
        return self, self.first, self.second, self.first, self

    def region(self, index, hidden):
        match index:
            case 0:
                return torch.tanh(hidden @ self.embed.T + self.scale + self.frozen)
            case 1 | 3:
                return torch.tanh(self.first.linear(hidden))
            case 2:
                return torch.tanh(self.second.linear(hidden))
            case 4:
                return (hidden @ self.embed).square().sum() * self.scale.sum() * 1e-2


def _inputs(rank, microbatch):
    generator = torch.Generator(device="cuda").manual_seed(100 * rank + microbatch)
    return torch.randn(3, _HIDDEN, device="cuda", generator=generator)


def _microbatch(stage, model, order, inputs, final, prefetch, dtype):
    regions = []
    hidden = inputs if dtype is None else inputs.to(dtype)
    for index, group in enumerate(order):
        region_inputs = hidden.detach().requires_grad_(index > 0)
        # As the program does, the prefetch follows the order that the first microbatch recorded.
        recorded = stage.order
        next_group = recorded[index + 1] if prefetch and recorded is not None and index + 1 < len(order) else None
        with stage.group_weights(group, next_group=next_group):
            hidden = model.region(index, region_inputs)
        regions.append((region_inputs, hidden))
    stage.record_order(order)

    backward = stage.backward(final=final)
    grad = None
    for index in reversed(range(len(order))):
        region_inputs, outputs = regions[index]
        with backward.region(index):
            torch.autograd.backward(outputs, grad)
        grad = region_inputs.grad


def _reference_step(reference, step, regions):
    reference.zero_grad()
    for rank in range(dist.get_world_size()):
        for microbatch in range(2):
            hidden = _inputs(rank, 10 * step + microbatch)
            for index in range(regions):
                hidden = reference.region(index, hidden)
            hidden.backward()


def _assert_grads_match(model, reference, tolerance):
    for (name, param), expected in zip(model.named_parameters(), reference.parameters(), strict=True):
        if not param.requires_grad:
            assert param.grad is None
        elif expected.grad is None:
            assert torch.equal(local_grad(param), torch.zeros_like(local_grad(param)))
        else:
            # The full gradient, because the shard of a rank can be empty.
            grad = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
            assert_angle_and_norm_close(grad, expected.grad, tol=tolerance, name=name)


@torch.no_grad()
def _sgd(model):
    for param in model.parameters():
        if param.grad is not None:
            param.sub_(1e-3 * param.grad)


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("prefetch", [False, True])
def test_regions_match_one_device(dist_ctx_factory, mesh, prefetch):
    dist_ctx = dist_ctx_factory(mesh)
    model = _Stage()
    reference = copy.deepcopy(model)
    place_hsdp(model, dist_ctx)
    runtime = make_runtime()
    runtime.register_stage(0, model, marked_parameter_groups(model))
    runtime.bind()
    try:
        stage = runtime.stage(0)
        order = tuple(stage.group_index(module) for module in model.region_modules())
        tolerance = grad_tolerance_for(torch.float32)
        # Several steps check that the gathers read the weights that the previous optimizer step wrote.
        for step in range(3):
            runtime.zero_grad()
            for microbatch in range(2):
                inputs = _inputs(dist.get_rank(), 10 * step + microbatch)
                _microbatch(stage, model, order, inputs, final=microbatch == 1, prefetch=prefetch, dtype=None)
            runtime.wait()
            _reference_step(reference, step, len(order))
            _assert_grads_match(model, reference, tolerance)
            _sgd(model)
            _sgd(reference)
    finally:
        runtime.unbind()
