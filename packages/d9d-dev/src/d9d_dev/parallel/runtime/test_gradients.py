import pytest
import torch
import torch.distributed as dist
from d9d.dist_context import DENSE_DOMAIN, DenseDim, DeviceMeshParameters
from torch import nn
from torch.distributed.tensor import DTensor, Shard

from d9d_dev.nn.helper.tolerances import forward_tolerance_for
from d9d_dev.parallel.runtime.helpers import (
    MESHES,
    distribute_parameters,
    local_grad,
    local_part,
    make_runtime,
    place_hsdp,
    run_one_region,
)


def _inputs(rank):
    return torch.arange(64, dtype=torch.float32, device="cuda").view(4, 16) * ((rank + 1) / 64)


class _MixedWeights(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.full = nn.Parameter(torch.full((out_features, 16), 0.01, device="cuda", dtype=torch.float32))
        self.low = nn.Parameter(torch.full((out_features, 16), 0.01, device="cuda", dtype=torch.bfloat16))

    def forward(self, inputs):
        return inputs @ self.full.T + (inputs.bfloat16() @ self.low.T).float()


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("out_features", [3, 17], ids=["fewer_rows_than_ranks", "uneven_rows"])
def test_gradients_match_one_device(dist_ctx_factory, mesh, out_features):
    dist_ctx = dist_ctx_factory(mesh)
    model = _MixedWeights(out_features)
    reference = _MixedWeights(out_features)
    place_hsdp(model, dist_ctx)
    for param in (*model.parameters(), *reference.parameters()):
        param.grad_dtype = torch.float32
    runtime = make_runtime()
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        runtime.zero_grad()
        run_one_region(runtime, lambda: model(_inputs(dist.get_rank())).square().sum().backward())
        for rank in range(dist.get_world_size()):
            reference(_inputs(rank)).square().sum().backward()
        # The bf16 half of the model rounds its products, so the gradients match only to bf16 tolerances.
        tolerance = forward_tolerance_for(torch.bfloat16)
        for param, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(
                local_grad(param), local_part(param, expected.grad), atol=tolerance.atol, rtol=tolerance.rtol
            )
    finally:
        runtime.unbind()


def _hooked_linear(dist_ctx):
    model = nn.Linear(16, 17, bias=False, device="cuda")
    with torch.no_grad():
        model.weight.fill_(0.01)
    reference = nn.Linear(16, 17, bias=False, device="cuda")
    reference.load_state_dict(model.state_dict())
    place_hsdp(model, dist_ctx)
    runtime = make_runtime()
    runtime.register_stage(0, model)
    return model, reference, runtime


def _step(runtime, model):
    runtime.zero_grad()
    run_one_region(runtime, lambda: model(_inputs(dist.get_rank())).square().sum().backward())


def _double(param):
    param.grad.mul_(2)


def _replace_by_double(param):
    param.grad = param.grad * 2


def _drop(param):
    param.grad = None


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("hook", [_double, _replace_by_double, _drop], ids=["in_place", "replacing", "dropping"])
def test_post_accumulate_hooks_see_the_reduced_shard(dist_ctx_factory, mesh, hook):
    model, reference, runtime = _hooked_linear(dist_ctx_factory(mesh))
    seen = []
    model.weight.register_post_accumulate_grad_hook(seen.append)
    model.weight.register_post_accumulate_grad_hook(hook)
    runtime.bind()
    try:
        if hook is _drop:
            with pytest.raises(ValueError, match="to None"):
                _step(runtime, model)
            return
        _step(runtime, model)
        # The hooks run once on the sharded parameter, after the gradient is summed over all ranks. A
        # replaced gradient lands in the bucket, so scale_grads() still reaches it.
        assert seen == [model.weight]
        runtime.scale_grads(torch.tensor(0.5, device="cuda"))
        for rank in range(dist.get_world_size()):
            reference(_inputs(rank)).square().sum().backward()
        torch.testing.assert_close(local_grad(model.weight), local_part(model.weight, reference.weight.grad))
    finally:
        runtime.unbind()


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("removed", [False, True])
def test_tensor_hooks_see_the_local_gradient(dist_ctx_factory, mesh, removed):
    model, reference, runtime = _hooked_linear(dist_ctx_factory(mesh))
    seen = []
    handle = model.weight.register_hook(lambda grad: seen.append(grad.clone()))
    if removed:
        handle.remove()
    runtime.bind()
    try:
        _step(runtime, model)
        if removed:
            assert seen == []
            return
        # Tensor hooks see the gradient of each backward pass of this rank, before any reduction.
        reference(_inputs(dist.get_rank())).square().sum().backward()
        assert len(seen) == 1
        assert not isinstance(seen[0], DTensor)
        torch.testing.assert_close(seen[0], reference.weight.grad)
    finally:
        runtime.unbind()


class _ConditionalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(4, 4, bias=False, device="cuda")
        self.head = nn.Linear(4, 4, bias=False, device="cuda")
        self.use_head = True

    def forward(self, inputs):
        outputs = self.body(inputs)
        return self.head(outputs) if self.use_head else outputs


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", MESHES)
def test_unused_parameters_get_zero_gradients(dist_ctx_factory, mesh):
    dist_ctx = dist_ctx_factory(mesh)
    model = _ConditionalHead()
    reference = _ConditionalHead()
    reference.load_state_dict(model.state_dict())
    place_hsdp(model, dist_ctx)
    runtime = make_runtime()
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        # The first step uses the head on rank 0 only, the second step on no rank. As in the gradient synchronizer,
        # a parameter gets the sum of the ranks that used it, and zeros if none did.
        for uses_head in ([True] + [False] * 7, [False] * 8):
            model.use_head = uses_head[dist.get_rank()]
            runtime.zero_grad()
            run_one_region(runtime, lambda: model(torch.ones(2, 4, device="cuda")).sum().backward())

            reference.zero_grad()
            for use_head in uses_head:
                reference.use_head = use_head
                reference(torch.ones(2, 4, device="cuda")).sum().backward()
            for param, expected in zip(model.parameters(), reference.parameters(), strict=True):
                full = torch.zeros_like(expected) if expected.grad is None else expected.grad
                torch.testing.assert_close(local_grad(param), local_part(param, full))
    finally:
        runtime.unbind()


@pytest.mark.distributed
def test_layer_sharded_parameter_keeps_the_rows_of_each_rank(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=8))
    mesh = dist_ctx.mesh_for(DENSE_DOMAIN)[DenseDim.DP_REPLICATE]
    model = nn.Linear(4, 8 * 4, bias=False, device="cuda")
    # Shard(0) outside the data shard dimensions: each rank holds its own rows, as expert parallelism does.
    distribute_parameters(model, mesh, (Shard(0),))
    runtime = make_runtime()
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        runtime.zero_grad()

        def _backward():
            if dist.get_rank() == 0:
                model(torch.ones(2, 4, device="cuda")).sum().backward()

        run_one_region(runtime, _backward)
        # Only rank 0 used its rows, and the runtime leaves the rows of a layer-sharded parameter to their rank.
        expected = 2.0 if dist.get_rank() == 0 else 0.0
        torch.testing.assert_close(local_grad(model.weight), torch.full_like(model.weight.to_local(), expected))
    finally:
        runtime.unbind()


@pytest.mark.local
def test_post_accumulate_hook_can_remove_itself():
    model = nn.Linear(4, 4, bias=False, device="cuda")
    calls = []
    handles = []

    def first(_param):
        calls.append("first")
        handles[0].remove()

    handles.append(model.weight.register_post_accumulate_grad_hook(first))
    model.weight.register_post_accumulate_grad_hook(lambda _param: calls.append("second"))
    runtime = make_runtime()
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        for _ in range(2):
            runtime.zero_grad()
            run_one_region(runtime, lambda: model(torch.ones(2, 4, device="cuda")).sum().backward())
        assert calls == ["first", "second", "second"]
    finally:
        runtime.unbind()


@pytest.mark.local
@pytest.mark.filterwarnings("ignore:Using backward\\(\\) with create_graph=True")
def test_backward_that_replaces_gradients_is_rejected():
    model = nn.Linear(4, 4, bias=False, device="cuda")
    runtime = make_runtime()
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        runtime.zero_grad()
        # With create_graph=True, autograd replaces the gradient instead of adding into the buffer of the runtime.
        with pytest.raises(ValueError, match=r"\(Linear.weight\) was replaced during the backward pass"):
            run_one_region(
                runtime, lambda: model(torch.ones(2, 4, device="cuda")).square().sum().backward(create_graph=True)
            )
    finally:
        runtime.unbind()


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("after_reduction", [False, True])
def test_zero_grad_recovers_from_a_failed_backward(dist_ctx_factory, mesh, after_reduction):
    dist_ctx = dist_ctx_factory(mesh)
    model = nn.Linear(16, 17, bias=False, device="cuda")
    with torch.no_grad():
        model.weight.fill_(0.01)
    reference = nn.Linear(16, 17, bias=False, device="cuda")
    reference.load_state_dict(model.state_dict())
    place_hsdp(model, dist_ctx)
    runtime = make_runtime()
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        runtime.zero_grad()
        stage = runtime.stage(0)
        stage.record_order((0,))
        backward = stage.backward(final=True)
        with pytest.raises(RuntimeError, match="Injected"):
            with backward.region(0):
                model(_inputs(dist.get_rank())).square().sum().backward()
                if not after_reduction:
                    raise RuntimeError("Injected failure inside the region.")
            raise RuntimeError("Injected failure before wait().")

        runtime.zero_grad()
        run_one_region(runtime, lambda: model(_inputs(dist.get_rank())).square().sum().backward())
        for rank in range(dist.get_world_size()):
            reference(_inputs(rank)).square().sum().backward()
        torch.testing.assert_close(local_grad(model.weight), local_part(model.weight, reference.weight.grad))
    finally:
        runtime.unbind()
