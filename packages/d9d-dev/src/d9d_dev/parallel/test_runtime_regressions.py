import pytest
import torch
import torch.distributed as dist
from d9d.dist_context import DENSE_DOMAIN, DenseDim, DenseDomainConfig, DeviceMeshParameters
from d9d.parallel import distribute_parameters
from d9d.pipelining.api import mark_window
from torch import nn
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from d9d_dev.nn.helper.tolerances import forward_tolerance_for
from d9d_dev.parallel.runtime_step import MESHES, make_runtime, place_hsdp


class _MixedWeights(nn.Module):
    def __init__(self, out_features=17):
        super().__init__()
        self.full = nn.Parameter(torch.full((out_features, 16), 0.01, device="cuda", dtype=torch.float32))
        self.low = nn.Parameter(torch.full((out_features, 16), 0.01, device="cuda", dtype=torch.bfloat16))

    def forward(self, inputs):
        return inputs @ self.full.T + (inputs.bfloat16() @ self.low.T).float()


def _inputs(rank):
    return torch.arange(64, dtype=torch.float32, device="cuda").view(4, 16) * ((rank + 1) / 64)


def _distribute(model, dist_ctx):
    place_hsdp(model, dist_ctx)


def _backward(runtime, model, inputs):
    with runtime.group_weights(0, 0):
        loss = model(inputs).square().sum()
    with runtime.group_weights(0, 0, prepare=runtime.reduction_plan(0, (0,), final=True).before_region.get(0, ())):
        loss.backward()
    plan = runtime.reduction_plan(0, (0,), final=True)
    for bucket_ids in plan.after_region.values():
        for bucket_id in bucket_ids:
            runtime.reduce(plan, bucket_id)
    runtime.wait()


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("out_features", [3, 17])
def test_mixed_storage_dtypes_keep_compute_dtypes(dist_ctx_factory, mesh, out_features):
    dist_ctx = dist_ctx_factory(mesh)
    model = _MixedWeights(out_features)
    reference = _MixedWeights(out_features)
    _distribute(model, dist_ctx)
    for param in (*model.parameters(), *reference.parameters()):
        param.grad_dtype = torch.float32
    runtime = make_runtime(memory_budget_bytes=1)
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        _backward(runtime, model, _inputs(dist.get_rank()))
        for rank in range(dist.get_world_size()):
            reference(_inputs(rank)).square().sum().backward()
        for param, expected in zip(model.parameters(), reference.parameters(), strict=True):
            assert isinstance(param, DTensor)
            local = distribute_tensor(expected.grad, param.device_mesh, param.placements).to_local()
            # The bf16 half of the model rounds its products, so the gradients match only to bf16 tolerances.
            tolerance = forward_tolerance_for(torch.bfloat16)
            torch.testing.assert_close(param.grad.to_local(), local, atol=tolerance.atol, rtol=tolerance.rtol)
    finally:
        runtime.unbind()


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", MESHES)
def test_post_accumulate_hook_sees_reduced_sharded_gradient(dist_ctx_factory, mesh):
    dist_ctx = dist_ctx_factory(mesh)
    model = nn.Linear(16, 17, bias=False, device="cuda")
    with torch.no_grad():
        model.weight.fill_(0.01)
    reference = nn.Linear(16, 17, bias=False, device="cuda")
    reference.load_state_dict(model.state_dict())
    _distribute(model, dist_ctx)
    seen = []

    def _double_gradient(param):
        seen.append(param)
        param.grad.mul_(2)

    model.weight.register_post_accumulate_grad_hook(_double_gradient)
    runtime = make_runtime(memory_budget_bytes=1)
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        _backward(runtime, model, _inputs(dist.get_rank()))
        for rank in range(dist.get_world_size()):
            reference(_inputs(rank)).square().sum().backward()
        weight = model.weight
        # As in FSDP2, the hook runs once on the sharded parameter, after the gradient is summed over all ranks.
        assert len(seen) == 1
        assert seen[0] is weight
        expected = distribute_tensor(2 * reference.weight.grad, weight.device_mesh, weight.placements).to_local()
        torch.testing.assert_close(weight.grad.to_local(), expected)
    finally:
        runtime.unbind()


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", MESHES)
def test_tensor_hook_sees_local_unsharded_gradient(dist_ctx_factory, mesh):
    dist_ctx = dist_ctx_factory(mesh)
    model = nn.Linear(16, 17, bias=False, device="cuda")
    with torch.no_grad():
        model.weight.fill_(0.01)
    reference = nn.Linear(16, 17, bias=False, device="cuda")
    reference.load_state_dict(model.state_dict())
    _distribute(model, dist_ctx)
    seen = []
    model.weight.register_hook(lambda grad: seen.append(grad.clone()))
    runtime = make_runtime(memory_budget_bytes=1)
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        _backward(runtime, model, _inputs(dist.get_rank()))
        reference(_inputs(dist.get_rank())).square().sum().backward()
        assert len(seen) == 1
        assert not isinstance(seen[0], DTensor)
        torch.testing.assert_close(seen[0], reference.weight.grad)
    finally:
        runtime.unbind()


def _hooked_linear(dist_ctx):
    model = nn.Linear(16, 17, bias=False, device="cuda")
    with torch.no_grad():
        model.weight.fill_(0.01)
    reference = nn.Linear(16, 17, bias=False, device="cuda")
    reference.load_state_dict(model.state_dict())
    _distribute(model, dist_ctx)
    for rank in range(dist.get_world_size()):
        reference(_inputs(rank)).square().sum().backward()
    return model, reference


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", MESHES)
def test_replaced_gradient_returns_to_the_runtime_buffer(dist_ctx_factory, mesh):
    dist_ctx = dist_ctx_factory(mesh)
    model, reference = _hooked_linear(dist_ctx)

    def _replace(param):
        param.grad = param.grad * 2

    model.weight.register_post_accumulate_grad_hook(_replace)
    runtime = make_runtime(memory_budget_bytes=1)
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        _backward(runtime, model, _inputs(dist.get_rank()))
        # The value of the new tensor lands in the shard, so scale_grads() still reaches it.
        runtime.scale_grads(torch.tensor(0.5, device="cuda"))
        weight = model.weight
        expected = distribute_tensor(reference.weight.grad, weight.device_mesh, weight.placements).to_local()
        torch.testing.assert_close(weight.grad.to_local(), expected)
    finally:
        runtime.unbind()


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", MESHES)
def test_hook_cannot_drop_the_gradient(dist_ctx_factory, mesh):
    dist_ctx = dist_ctx_factory(mesh)
    model, _ = _hooked_linear(dist_ctx)

    def _drop(param):
        param.grad = None

    model.weight.register_post_accumulate_grad_hook(_drop)
    runtime = make_runtime(memory_budget_bytes=1)
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        with pytest.raises(ValueError, match="to None"):
            _backward(runtime, model, _inputs(dist.get_rank()))
    finally:
        runtime.unbind()


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", MESHES)
def test_wait_rejects_a_replaced_gradient(dist_ctx_factory, mesh):
    dist_ctx = dist_ctx_factory(mesh)
    model, _ = _hooked_linear(dist_ctx)
    runtime = make_runtime(memory_budget_bytes=1)
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        with runtime.group_weights(0, 0, prepare=runtime.reduction_plan(0, (0,), final=True).before_region.get(0, ())):
            model(_inputs(dist.get_rank())).square().sum().backward()
        plan = runtime.reduction_plan(0, (0,), final=True)
        for bucket_ids in plan.after_region.values():
            for bucket_id in bucket_ids:
                runtime.reduce(plan, bucket_id)
        model.weight.grad = torch.empty_like(model.weight.grad)
        with pytest.raises(ValueError, match="was replaced"):
            runtime.wait()
    finally:
        runtime.unbind()


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", MESHES)
def test_removed_tensor_hook_stops_running(dist_ctx_factory, mesh):
    dist_ctx = dist_ctx_factory(mesh)
    model, _ = _hooked_linear(dist_ctx)
    calls = []
    handle = model.weight.register_hook(lambda grad: calls.append(1))
    runtime = make_runtime(memory_budget_bytes=1)
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        handle.remove()
        _backward(runtime, model, _inputs(dist.get_rank()))
        assert calls == []
    finally:
        runtime.unbind()


@pytest.mark.local
def test_compute_dtype_gives_parameters_without_gather_their_own_copy():
    model = nn.Linear(16, 17, device="cuda")
    expected = {name: param.detach().to(torch.bfloat16) for name, param in model.named_parameters()}
    runtime = make_runtime(compute_dtype=torch.bfloat16)
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        with runtime.group_weights(0, 0):
            for name, param in model.named_parameters():
                assert torch.equal(param, expected[name])
        for param in model.parameters():
            assert param.dtype == torch.float32
    finally:
        runtime.unbind()


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", MESHES)
def test_rebind_refreshes_checkpoint_weights(dist_ctx_factory, mesh):
    dist_ctx = dist_ctx_factory(mesh)
    model = nn.Linear(16, 17, bias=False, device="cuda")
    _distribute(model, dist_ctx)
    runtime = make_runtime(compute_dtype=torch.bfloat16, memory_budget_bytes=2**20)
    runtime.register_stage(0, model)
    inputs = torch.ones((1, 16), device="cuda", dtype=torch.bfloat16)
    runtime.bind(training=False)
    with torch.inference_mode(), runtime.group_weights(0, 0):
        model(inputs)
    runtime.unbind()
    runtime.bind(training=False)
    try:
        # A checkpoint can load after the new compute buffers have been constructed.
        with torch.no_grad():
            model.weight.to_local().fill_(2)
        with torch.inference_mode(), runtime.group_weights(0, 0):
            actual = model(inputs)
        torch.testing.assert_close(actual, torch.full_like(actual, 32))
    finally:
        runtime.unbind()


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("training", [False, True])
def test_checkpoint_invalidates_frozen_compute_weights(dist_ctx_factory, mesh, training):
    dist_ctx = dist_ctx_factory(mesh)
    model = nn.Linear(16, 17, bias=False, device="cuda").requires_grad_(False)
    _distribute(model, dist_ctx)
    runtime = make_runtime(compute_dtype=torch.bfloat16, memory_budget_bytes=2**20)
    runtime.register_stage(0, model)
    runtime.bind(training=training)
    inputs = torch.ones((1, 16), device="cuda", dtype=torch.bfloat16)
    try:
        with torch.no_grad(), runtime.group_weights(0, 0):
            model(inputs)
        runtime.invalidate_weights()
        with torch.no_grad():
            model.weight.to_local().fill_(2)
        with torch.no_grad(), runtime.group_weights(0, 0):
            actual = model(inputs)
        torch.testing.assert_close(actual, torch.full_like(actual, 32))
    finally:
        runtime.unbind()


@pytest.mark.local
def test_aliases_in_one_module_share_the_unsharded_parameter():
    module = nn.Module()
    module.first = nn.Parameter(torch.ones(4, device="cuda"))
    module.second = module.first
    runtime = make_runtime(compute_dtype=torch.bfloat16)
    runtime.register_stage(0, module)
    runtime.bind()
    try:
        with runtime.group_weights(0, 0):
            assert module.first is module.second
            assert module.first.dtype == torch.bfloat16
    finally:
        runtime.unbind()


@pytest.mark.distributed
def test_runtime_rejects_shards_along_another_dimension_of_a_gather_dim(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=8)))
    model = nn.Linear(16, 16, bias=False, device="cuda")
    distribute_parameters(model, dist_ctx.mesh_for(DENSE_DOMAIN)[DenseDim.DP_CP_SHARD], (Shard(1),))
    runtime = make_runtime()
    runtime.register_stage(0, model)
    with pytest.raises(ValueError, match="cannot gather"):
        runtime.bind()


@pytest.mark.local
def test_invalidate_weights_before_bind_refreshes_the_first_step():
    model = nn.Linear(4, 4, bias=False, device="cuda")
    runtime = make_runtime(compute_dtype=torch.bfloat16)
    runtime.invalidate_weights()
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        with torch.no_grad():
            model.weight.fill_(2)
        with runtime.group_weights(0, 0):
            assert torch.equal(model.weight, torch.full_like(model.weight, 2))
    finally:
        runtime.unbind()


@pytest.mark.local
def test_runtime_rejects_parameters_tied_across_stages():
    first = nn.Linear(4, 4)
    second = nn.Linear(4, 4)
    second.weight = first.weight
    runtime = make_runtime()
    runtime.register_stage(0, first)
    runtime.register_stage(1, second)
    with pytest.raises(ValueError, match="Place tied parameters on one stage"):
        runtime.bind()


@pytest.mark.local
def test_bind_converts_only_non_persistent_float_buffers():
    module = nn.Linear(4, 4, device="cuda")
    module.register_buffer("cache", torch.ones(4, device="cuda"), persistent=False)
    module.register_buffer("state", torch.ones(4, device="cuda"), persistent=True)
    module.register_buffer("counts", torch.ones(4, dtype=torch.int64, device="cuda"), persistent=False)
    # A table that several layers share, such as the rotary table of a decoder.
    module.child = nn.Module()
    module.child.register_buffer("cache", module.cache, persistent=False)
    runtime = make_runtime(compute_dtype=torch.bfloat16)
    runtime.register_stage(0, module)
    runtime.bind()
    try:
        assert module.cache.dtype == torch.bfloat16
        assert module.child.cache is module.cache
        assert module.state.dtype == torch.float32
        assert module.counts.dtype == torch.int64
        assert set(module.state_dict()) == {"weight", "bias", "state"}
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


@pytest.mark.local
@pytest.mark.parametrize(
    ("weight_decay", "use_head"),
    [pytest.param(0.1, [False, False], id="decay_unused"), pytest.param(0.0, [True, False], id="momentum_unused")],
)
def test_unused_parameters_get_no_gradient(weight_decay, use_head):
    model = _ConditionalHead()
    reference = _ConditionalHead()
    reference.load_state_dict(model.state_dict())
    runtime = make_runtime()
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=weight_decay)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.1, weight_decay=weight_decay)

        for value in use_head:
            model.use_head = value
            runtime.zero_grad()
            _backward(runtime, model, torch.ones(2, 4, device="cuda"))
            optimizer.step()

            reference.use_head = value
            reference_optimizer.zero_grad(set_to_none=True)
            reference(torch.ones(2, 4, device="cuda")).square().sum().backward()
            reference_optimizer.step()

        assert model.head.weight.grad is None
        for param, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(param, expected)
    finally:
        runtime.unbind()


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", MESHES)
def test_parameter_used_on_some_ranks_keeps_its_gradient(dist_ctx_factory, mesh):
    dist_ctx = dist_ctx_factory(mesh)
    model = _ConditionalHead()
    _distribute(model, dist_ctx)
    model.use_head = dist.get_rank() == 0
    runtime = make_runtime(memory_budget_bytes=1)
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        runtime.zero_grad()
        _backward(runtime, model, torch.ones(2, 4, device="cuda"))
        assert model.head.weight.grad is not None
        assert model.body.weight.grad is not None
        runtime.zero_grad()
        model.use_head = False
        _backward(runtime, model, torch.ones(2, 4, device="cuda"))
        assert model.head.weight.grad is None
    finally:
        runtime.unbind()


@pytest.mark.distributed
def test_part_of_a_layer_sharded_parameter_used_on_one_rank_keeps_all_gradients(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=8))
    mesh = dist_ctx.mesh_for(DENSE_DOMAIN)[DenseDim.DP_REPLICATE]
    model = nn.Linear(4, 8 * 4, bias=False, device="cuda")
    # Shard(0) outside the data shard dimensions: each rank holds its own rows, as expert parallelism does.
    distribute_parameters(model, mesh, (Shard(0),))
    runtime = make_runtime(memory_budget_bytes=1)
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        runtime.zero_grad()
        with runtime.group_weights(0, 0, prepare=runtime.reduction_plan(0, (0,), final=True).before_region.get(0, ())):
            if dist.get_rank() == 0:
                model(torch.ones(2, 4, device="cuda")).sum().backward()
        plan = runtime.reduction_plan(0, (0,), final=True)
        for bucket_ids in plan.after_region.values():
            for bucket_id in bucket_ids:
                runtime.reduce(plan, bucket_id)
        runtime.wait()

        # The rows of the other ranks got no tokens, so their gradient is zero, not missing.
        assert model.weight.grad is not None
        expected = 2.0 if dist.get_rank() == 0 else 0.0
        torch.testing.assert_close(model.weight.grad.to_local(), torch.full_like(model.weight.to_local(), expected))
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
        runtime.zero_grad()
        _backward(runtime, model, torch.ones(2, 4, device="cuda"))
        assert calls == ["first", "second"]
    finally:
        runtime.unbind()


@pytest.mark.local
def test_runtime_lifecycle_is_explicit():
    runtime = make_runtime()
    runtime.register_stage(0, nn.Linear(4, 4, device="cuda"))
    with pytest.raises(ValueError, match="already registered"):
        runtime.register_stage(0, nn.Linear(4, 4, device="cuda"))
    for call in (
        lambda: runtime.reduction_plan(0, (0,), final=True),
        lambda: runtime.scale_grads(torch.ones(())),
        runtime.wait,
    ):
        with pytest.raises(ValueError, match="Call bind\\(\\) first"):
            call()
    runtime.bind()
    try:
        with pytest.raises(ValueError, match="Call unbind\\(\\) first"):
            runtime.register_stage(1, nn.Linear(4, 4, device="cuda"))
    finally:
        runtime.unbind()
    runtime.unbind()


class _TiedGroups(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = mark_window(nn.Linear(4, 4, bias=False, device="cuda"))
        self.second = mark_window(nn.Linear(4, 4, bias=False, device="cuda"))
        self.second.weight = self.first.weight

    async def forward_steps(self, steps, inputs, shared):
        hidden = await steps.call(self.first, inputs)
        return await steps.call(self.second, hidden)


@pytest.mark.local
def test_tied_parameter_sees_optimizer_writes_in_every_group():
    model = _TiedGroups()
    runtime = make_runtime(compute_dtype=torch.bfloat16)
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        with torch.no_grad():
            model.first.weight.fill_(2)
        for module in (model.first, model.second):
            with runtime.group_weights(0, runtime.group_index(0, module)):
                assert model.first.weight is model.second.weight
                assert torch.equal(module.weight, torch.full_like(module.weight, 2))
    finally:
        runtime.unbind()


class _ParameterAliasGroup(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, value, constants):
        return value * self.scale


class _ParameterAliasStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(4, device="cuda"))
        self.alias = mark_window(_ParameterAliasGroup(self.scale))

    async def forward_steps(self, steps, inputs, shared):
        return await steps.call(self.alias, inputs, shared)


@pytest.mark.local
def test_group_module_keeps_the_original_parameter_alias():
    model = _ParameterAliasStage()
    original = model.scale
    runtime = make_runtime(compute_dtype=torch.bfloat16)
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        alias = model.alias
        with runtime.group_weights(0, 0):
            assert model.scale is alias.scale
            assert model.scale.dtype == torch.bfloat16
        assert model.scale is original
        assert alias.scale is original
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
    _distribute(model, dist_ctx)
    runtime = make_runtime()
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        runtime.zero_grad()
        with pytest.raises(RuntimeError, match="Injected"):
            with runtime.group_weights(
                0, 0, prepare=runtime.reduction_plan(0, (0,), final=True).before_region.get(0, ())
            ):
                model(_inputs(dist.get_rank())).square().sum().backward()
            if after_reduction:
                plan = runtime.reduction_plan(0, (0,), final=True)
                for bucket_ids in plan.after_region.values():
                    for bucket_id in bucket_ids:
                        runtime.reduce(plan, bucket_id)
            raise RuntimeError("Injected failure of a backward pass.")

        runtime.zero_grad()
        _backward(runtime, model, _inputs(dist.get_rank()))
        for rank in range(dist.get_world_size()):
            reference(_inputs(rank)).square().sum().backward()
        weight = model.weight
        expected = distribute_tensor(reference.weight.grad, weight.device_mesh, weight.placements).to_local()
        torch.testing.assert_close(weight.grad.to_local(), expected)
    finally:
        runtime.unbind()
