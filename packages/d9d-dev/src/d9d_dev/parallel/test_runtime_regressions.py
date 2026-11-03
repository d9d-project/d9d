from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from d9d.dist_context import DENSE_DOMAIN
from d9d.parallel import parallelize_hsdp
from d9d.parallel.memory import MemoryLedger
from d9d.parallel.runtime.parameter import ParameterHandle, build_handles, convert_buffers
from d9d.parallel.runtime.plan import BucketPlan, classify_axes, minimum_stage_budget, plan_buckets, shard_sizes
from d9d.pipelining.api import StageSegment
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from d9d_dev.parallel.runtime_step import make_runtime
from d9d_dev.parallel.test_runtime_features import _MESHES


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
    mesh = dist_ctx.mesh_for(DENSE_DOMAIN)["dp_replicate", "dp_cp_shard", "cp_replicate"]
    parallelize_hsdp(model, mesh=mesh)


def _backward(runtime, model, inputs):
    with runtime.segment_weights(0, 0, writes_grads=False):
        loss = model(inputs).square().sum()
    with runtime.segment_weights(0, 0, writes_grads=True):
        loss.backward()
    plan = runtime.reduction_plan(0, final=True)
    for bucket_ids in plan.after_segment.values():
        for bucket_id in bucket_ids:
            runtime.reduce(bucket_id, plan.final)
    runtime.wait()


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", _MESHES)
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
            torch.testing.assert_close(param.grad.to_local(), local, rtol=1e-4, atol=1e-4)
    finally:
        runtime.unbind()


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", _MESHES)
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
@pytest.mark.parametrize("mesh", _MESHES)
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
@pytest.mark.parametrize("mesh", _MESHES)
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
@pytest.mark.parametrize("mesh", _MESHES)
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
@pytest.mark.parametrize("mesh", _MESHES)
def test_wait_rejects_a_replaced_gradient(dist_ctx_factory, mesh):
    dist_ctx = dist_ctx_factory(mesh)
    model, _ = _hooked_linear(dist_ctx)
    runtime = make_runtime(memory_budget_bytes=1)
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        _backward(runtime, model, _inputs(dist.get_rank()))
        model.weight.grad = model.weight.grad.clone()
        with pytest.raises(ValueError, match="was replaced"):
            runtime.wait()
    finally:
        runtime.unbind()


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", _MESHES)
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
def test_own_copies_are_made_only_for_parameters_without_gather():
    model = nn.Linear(16, 17, device="cuda")
    runtime = make_runtime(compute_dtype=torch.bfloat16)
    runtime.register_stage(0, model)
    runtime.bind()
    try:
        for handle in runtime._stages[0].handles[0]:
            assert not handle.unsharded.is_meta
            assert handle.unsharded.dtype == torch.bfloat16
            torch.testing.assert_close(handle.unsharded.float(), handle.sharded.data, rtol=1e-2, atol=1e-2)
    finally:
        runtime.unbind()


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", _MESHES)
def test_rebind_refreshes_checkpoint_weights(dist_ctx_factory, mesh):
    dist_ctx = dist_ctx_factory(mesh)
    model = nn.Linear(16, 17, bias=False, device="cuda")
    _distribute(model, dist_ctx)
    runtime = make_runtime(compute_dtype=torch.bfloat16, memory_budget_bytes=2**20)
    runtime.register_stage(0, model)
    inputs = torch.ones((1, 16), device="cuda", dtype=torch.bfloat16)
    runtime.bind(training=False)
    with torch.inference_mode(), runtime.segment_weights(0, 0, writes_grads=False):
        model(inputs)
    runtime.unbind()
    runtime.bind(training=False)
    try:
        # A checkpoint can load after the new compute buffers have been constructed.
        with torch.no_grad():
            model.weight.to_local().fill_(2)
        with torch.inference_mode(), runtime.segment_weights(0, 0, writes_grads=False):
            actual = model(inputs)
        torch.testing.assert_close(actual, torch.full_like(actual, 32))
    finally:
        runtime.unbind()


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", _MESHES)
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
        with torch.no_grad(), runtime.segment_weights(0, 0, writes_grads=False):
            model(inputs)
        runtime.invalidate_weights()
        with torch.no_grad():
            model.weight.to_local().fill_(2)
        with torch.no_grad(), runtime.segment_weights(0, 0, writes_grads=False):
            actual = model(inputs)
        torch.testing.assert_close(actual, torch.full_like(actual, 32))
    finally:
        buffers = [bucket.weights for bucket in runtime._stages[0].buckets if bucket.weights is not None]
        runtime.unbind()
        assert all(not buffer.allocated for buffer in buffers)


@pytest.mark.local
def test_handles_preserve_same_module_parameter_aliases():
    module = nn.Module()
    module.first = nn.Parameter(torch.ones(4))
    module.second = module.first
    segment = StageSegment(name="aliases", modules=(module,), call=lambda value, _: value)
    handles = build_handles((segment,), compute_dtype=None, training=True)
    assert len(handles[0]) == 1
    assert handles[0][0].owners == [(module, "first"), (module, "second")]


@pytest.mark.local
def test_empty_parameter_shard_sizes():
    parameter = nn.Parameter(torch.empty((0, 16)))
    assert shard_sizes(parameter, 8) == [0] * 8


def _placed(names, placements):
    return Mock(device_mesh=Mock(mesh_dim_names=names), placements=placements)


_GATHER_DIMS = frozenset({"dp_shard", "dp_cp_shard"})


@pytest.mark.local
def test_axis_roles_follow_the_gather_dims():
    roles = classify_axes(_placed(("dp_replicate", "dp_cp_shard"), (Replicate(), Shard(0))), _GATHER_DIMS)
    assert roles.gather == "dp_cp_shard"
    assert roles.replicas == ("dp_replicate",)


@pytest.mark.local
def test_shards_outside_the_gather_dims_stay_with_the_layer():
    roles = classify_axes(_placed(("ep_replicate", "ep_shard"), (Replicate(), Shard(0))), _GATHER_DIMS)
    assert roles.gather is None
    assert roles.replicas == ("ep_replicate",)
    roles = classify_axes(_placed(("tp",), (Shard(1),)), _GATHER_DIMS)
    assert roles.gather is None
    assert roles.replicas == ()


@pytest.mark.local
def test_runtime_rejects_gathering_layer_sharded_parameters():
    with pytest.raises(ValueError, match="along another dimension too"):
        classify_axes(_placed(("ep_shard", "dp_shard"), (Shard(0), Shard(0))), _GATHER_DIMS)


@pytest.mark.local
def test_runtime_rejects_other_shards_along_gather_dims():
    with pytest.raises(ValueError, match="cannot gather"):
        classify_axes(_placed(("dp_shard",), (Shard(1),)), _GATHER_DIMS)


@pytest.mark.local
def test_buckets_follow_backward_order_and_size_limit():
    first = nn.Linear(256, 256)
    second = nn.Linear(256, 256)
    segments = (
        StageSegment(name="first", modules=(first,), call=lambda value, _: value),
        StageSegment(name="second", modules=(second,), call=lambda value, _: value),
    )
    handles = build_handles(segments, compute_dtype=None, training=True)
    # One weight takes 256 KiB of fp32 gradients, so a 1 MiB limit fits all four parameters.
    buckets = plan_buckets(handles, bucket_size_mb=1, gather_dims=_GATHER_DIMS)
    assert len(buckets) == 1
    assert [x.sharded for x in buckets[0].handles] == [second.bias, second.weight, first.bias, first.weight]
    assert buckets[0].segments == (0, 1)


@pytest.mark.local
def test_frozen_parameters_without_gather_get_no_bucket():
    module = nn.Linear(4, 4).requires_grad_(False)
    segment = StageSegment(name="frozen", modules=(module,), call=lambda value, _: value)
    handles = build_handles((segment,), compute_dtype=None, training=True)
    assert plan_buckets(handles, bucket_size_mb=1, gather_dims=_GATHER_DIMS) == []


@pytest.mark.local
def test_invalidate_weights_before_bind():
    runtime = make_runtime()
    runtime.invalidate_weights()
    assert runtime._stale


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
    assert runtime._reduce_stream is None


@pytest.mark.local
def test_memory_budget_covers_overlapping_tied_gradients():
    parameter = nn.Parameter(torch.ones(32))
    buckets = [
        BucketPlan(
            handles=[ParameterHandle([], parameter, parameter, True, False)],
            segments=(index, 15 - index),
            gather_group=Mock(),
            replica_groups=[],
            grad_dtype=torch.float32,
            trainable=True,
            numel=32,
            gradient_bytes=128,
        )
        for index in range(8)
    ]
    # All eight gradients stay live at the middle segment, besides the gather and its temporary buffers.
    assert minimum_stage_budget(buckets) >= 8 * 128 + 3 * 128


@pytest.mark.local
def test_memory_ledger_reports_live_allocation_pressure():
    ledger = MemoryLedger(10)
    ledger.reserve(8)
    with pytest.raises(ValueError, match="available memory budget"):
        ledger.reserve(4)


@pytest.mark.local
def test_memory_ledger_waits_for_retired_allocation():
    ledger = MemoryLedger(10)
    event = Mock()
    ledger.reserve(8)
    ledger.retire([event], 8)
    ledger.reserve(10)
    event.synchronize.assert_called_once_with()


@pytest.mark.local
def test_convert_buffers_keeps_persistent_state():
    module = nn.Module()
    module.register_buffer("cache", torch.ones(4), persistent=False)
    module.register_buffer("state", torch.ones(4), persistent=True)
    module.register_buffer("counts", torch.ones(4, dtype=torch.int64), persistent=False)

    convert_buffers(module, torch.bfloat16)

    assert module.cache.dtype == torch.bfloat16
    assert module.state.dtype == torch.float32
    assert module.counts.dtype == torch.int64
    assert set(module.state_dict()) == {"state"}
