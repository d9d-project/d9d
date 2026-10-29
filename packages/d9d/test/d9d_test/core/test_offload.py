import pytest
import torch
from d9d.core.dist_context import DENSE_DOMAIN, DeviceMeshParameters
from d9d.core.offload import OffloadContext, OnloadContext, offload_tensor, onload_tensor
from d9d.loop.component.model_stage_factory import TrackedModules
from d9d.pipelining.training.optimizer import PipelinedOptimizer
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard


@pytest.mark.local
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_offload_tensor_round_trip(dtype):
    if dtype.is_floating_point:
        original = torch.randn(16, 8).to(dtype)
    else:
        original = torch.randint(0, 100, (16, 8), dtype=dtype)
    snapshot = original.detach().clone()

    offloaded = offload_tensor(original, pin_memory=False)

    assert offloaded.host.device.type == "cpu"
    # the tensor's local storage now points at the host mirror
    assert original.data_ptr() == offloaded.host.data_ptr()

    onload_tensor(original, offloaded, device=torch.device("cpu"))

    assert original.dtype == dtype
    assert original.shape == snapshot.shape
    assert torch.equal(original, snapshot)
    # onload reallocates a fresh buffer; storage no longer aliases the host mirror
    assert original.data_ptr() != offloaded.host.data_ptr()


@pytest.mark.local
def test_tracked_modules_offload_guards(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())
    tracked = TrackedModules(dist_ctx, [nn.Linear(8, 4)], lambda _name, _tensor: True)
    offload_ctx = OffloadContext(dist_context=dist_ctx, pin_memory=False)

    assert not tracked.is_offloaded()
    with pytest.raises(RuntimeError, match="not offloaded"):
        tracked.onload(OnloadContext(dist_context=dist_ctx))

    tracked.offload(offload_ctx)
    assert tracked.is_offloaded()
    with pytest.raises(RuntimeError, match="already offloaded"):
        tracked.offload(offload_ctx)


@pytest.mark.distributed
@pytest.mark.parametrize("placement", [Shard(0), Replicate()])
def test_offload_dtensor_round_trip(dist_ctx_factory, placement):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    mesh = dist_ctx.mesh_for(DENSE_DOMAIN)["dp_replicate"]
    device = dist_ctx.current_device

    local = torch.randn(4, 8, device=device)
    original = DTensor.from_local(local, device_mesh=mesh, placements=(placement,))
    original_id = id(original)
    original_mesh = original.device_mesh
    original_placements = original.placements
    original_shape = original.shape

    offloaded = offload_tensor(original, pin_memory=True)

    assert offloaded.host.device.type == "cpu"
    assert offloaded.host.is_pinned()
    # the wrapper instance and its distributed metadata survive the offload
    assert id(original) == original_id
    assert original.device_mesh is original_mesh
    assert original.placements == original_placements
    assert original.shape == original_shape

    onload_tensor(original, offloaded, device=device)

    # same wrapper, same metadata, restored content
    assert id(original) == original_id
    assert original.device_mesh is original_mesh
    assert original.placements == original_placements
    assert original.shape == original_shape
    assert torch.equal(original.to_local(), local)


@pytest.mark.distributed
def test_tracked_modules_round_trip(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    device = dist_ctx.current_device

    module = nn.Sequential(nn.Linear(2048, 2048), nn.Linear(2048, 2048), nn.Linear(2048, 2048)).to(device)
    param_bytes = sum(p.numel() * p.element_size() for p in module.parameters())

    tracked = TrackedModules(dist_ctx, [module], lambda _name, _tensor: True)
    params = list(module.parameters())
    snapshots = [p.detach().clone() for p in params]

    mem_before = torch.cuda.memory_allocated(device)
    tracked.offload(OffloadContext(dist_context=dist_ctx, pin_memory=False))

    assert tracked.is_offloaded()
    assert all(p.device.type == "cpu" for p in module.parameters())
    # the GPU storage of the parameters must actually be released
    assert mem_before - torch.cuda.memory_allocated(device) >= 0.9 * param_bytes

    tracked.onload(OnloadContext(dist_context=dist_ctx))

    assert not tracked.is_offloaded()
    # parameter object identity is preserved across the round trip
    assert all(a is b for a, b in zip(module.parameters(), params, strict=True))
    for param, snapshot in zip(module.parameters(), snapshots, strict=True):
        assert param.device.type == "cuda"
        assert torch.equal(param.detach().cpu(), snapshot.cpu())

    # the module is still usable after waking up
    assert module(torch.randn(2, 2048, device=device)).shape == (2, 2048)


@pytest.mark.distributed
def test_pipelined_optimizer_round_trip(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    device = dist_ctx.current_device

    module = nn.Linear(64, 64).to(device)
    inner = torch.optim.AdamW(module.parameters())

    # populate the optimizer state with a step
    module(torch.randn(8, 64, device=device)).sum().backward()
    inner.step()

    optimizer = PipelinedOptimizer(None, [inner])
    snapshots = {
        id(param): {key: value.detach().clone() for key, value in state.items() if torch.is_tensor(value)}
        for param, state in inner.state.items()
    }
    state_ids = {
        id(param): {key: id(value) for key, value in state.items() if torch.is_tensor(value)}
        for param, state in inner.state.items()
    }

    assert not optimizer.is_offloaded()
    optimizer.offload(OffloadContext(dist_context=dist_ctx, pin_memory=False))

    assert optimizer.is_offloaded()
    for param, state in inner.state.items():
        for key, value in state.items():
            if torch.is_tensor(value):
                assert value.device.type == "cpu"
                # the optimizer-state dict entry is the same tensor object as before offload
                assert id(value) == state_ids[id(param)][key]

    optimizer.onload(OnloadContext(dist_context=dist_ctx))

    assert not optimizer.is_offloaded()
    for param, state in inner.state.items():
        for key, value in state.items():
            if torch.is_tensor(value):
                assert value.device.type == "cuda"
                # tensor object identity is preserved through the full round trip
                assert id(value) == state_ids[id(param)][key]
                assert torch.equal(value.detach().cpu(), snapshots[id(param)][key].cpu())

    # the optimizer remains usable after the round trip
    module(torch.randn(8, 64, device=device)).sum().backward()
    optimizer.step()
