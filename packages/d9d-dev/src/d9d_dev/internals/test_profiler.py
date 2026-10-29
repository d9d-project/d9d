import json
import tarfile
import threading

import pytest
import torch
from d9d.core.dist_context import REGULAR_DOMAIN, DeviceMeshParameters
from d9d.internals.profiling import Profiler


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("period_steps", "warmup_steps", "active_steps", "start_step", "simulate_steps", "will_write_at"),
    [
        (4, 1, 1, 0, 4, (4,)),
        (4, 1, 1, 0, 3, ()),
        (4, 1, 1, 0, 8, (4, 8)),
    ],
)
def test_e2e(
    dist_ctx_factory,
    shared_tmp_dir,
    period_steps,
    warmup_steps,
    active_steps,
    start_step,
    simulate_steps,
    will_write_at,
):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    profiler_wrapper = Profiler(
        save_dir=shared_tmp_dir,
        period_steps=period_steps,
        warmup_steps=warmup_steps,
        active_steps=active_steps,
        record_shapes=True,
        with_stack=True,
        dist_context=dist_ctx,
    )

    # We must run exactly 'period_steps' to complete one cycle and trigger the dump
    with profiler_wrapper.open(start_step=start_step) as prof:
        for _ in range(simulate_steps):
            x = torch.randn(100, 100, device="cuda")
            torch.matmul(x, x)
            torch.cuda.synchronize()

            prof.step()

    dist_ctx.wait_world()

    for written_step in will_write_at:
        step_dir = shared_tmp_dir / f"step_{written_step}"

        assert step_dir.is_dir()

        mesh = dist_ctx.mesh_for(REGULAR_DOMAIN)
        coord_str = "-".join(map(str, mesh.get_coordinate()))

        expected_filename_base = f"rank-{mesh.get_rank()}-coord-{coord_str}-trace"
        tar_path = step_dir / f"{expected_filename_base}.tar.gz"
        json_path = step_dir / f"{expected_filename_base}.json"

        assert not json_path.exists()
        assert tar_path.exists()
        # Check content of the tarball
        with tarfile.open(tar_path, "r:gz") as tar:
            expected_member_name = json_path.name
            try:
                member = tar.getmember(expected_member_name)
                assert member is not None
                assert member.size > 0
            except KeyError:
                pytest.fail(f"Tarball did not contain expected file: {expected_member_name}")


@pytest.mark.local
@pytest.mark.parametrize(
    ("period_steps", "warmup_steps", "active_steps", "start_step", "simulate_steps", "will_write_at"),
    [
        (4, 1, 1, 0, 4, (4,)),
        (4, 1, 1, 0, 3, ()),
        (4, 1, 1, 0, 8, (4, 8)),
    ],
)
def test_local_e2e(
    dist_ctx_factory, tmp_path, period_steps, warmup_steps, active_steps, start_step, simulate_steps, will_write_at
):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())
    profiler_wrapper = Profiler(
        save_dir=tmp_path,
        period_steps=period_steps,
        warmup_steps=warmup_steps,
        active_steps=active_steps,
        record_shapes=True,
        with_stack=True,
        dist_context=dist_ctx,
    )

    with profiler_wrapper.open(start_step=start_step) as prof:
        for _ in range(simulate_steps):
            # Simple workload
            x = torch.randn(10, 10, device="cuda")
            torch.matmul(x, x)
            torch.cuda.synchronize()

            prof.step()

    for written_step in will_write_at:
        step_dir = tmp_path / f"step_{written_step}"
        assert step_dir.is_dir()

        tar_path = step_dir / "trace.tar.gz"
        json_path = step_dir / "trace.json"

        # Ensure raw JSON is cleaned up and archive exists
        assert not json_path.exists()
        assert tar_path.exists()

        # Check content of the tarball
        with tarfile.open(tar_path, "r:gz") as tar:
            expected_member_name = json_path.name
            try:
                member = tar.getmember(expected_member_name)
                assert member is not None
                assert member.size > 0
            except KeyError:
                pytest.fail(f"Tarball did not contain expected file: {expected_member_name}")


@pytest.mark.local
def test_records_background_threads(dist_ctx_factory, tmp_path):
    profiler_wrapper = Profiler(
        save_dir=tmp_path,
        period_steps=2,
        warmup_steps=1,
        active_steps=1,
        record_shapes=True,
        with_stack=True,
        dist_context=dist_ctx_factory(DeviceMeshParameters()),
    )

    def background_work():
        with torch.profiler.record_function("background work"):
            torch.randn(10, 10).sum()

    with profiler_wrapper.open(start_step=0) as prof:
        for _ in range(2):
            worker = threading.Thread(target=background_work)
            worker.start()
            worker.join()
            prof.step()

    with tarfile.open(tmp_path / "step_2" / "trace.tar.gz", "r:gz") as tar:
        trace = json.load(tar.extractfile("trace.json"))
    events = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    step_threads = {event["tid"] for event in events if event["name"].startswith("ProfilerStep#")}
    background = [event for event in events if event["name"] == "background work"]

    assert background
    assert {event["tid"] for event in background}.isdisjoint(step_threads)


@pytest.mark.local
@pytest.mark.parametrize("enabled", [True, False])
def test_records_shapes_and_stacks_only_when_enabled(dist_ctx_factory, tmp_path, enabled):
    profiler_wrapper = Profiler(
        save_dir=tmp_path,
        period_steps=2,
        warmup_steps=1,
        active_steps=1,
        record_shapes=enabled,
        with_stack=enabled,
        dist_context=dist_ctx_factory(DeviceMeshParameters()),
    )

    with profiler_wrapper.open(start_step=0) as prof:
        for _ in range(2):
            torch.matmul(torch.randn(10, 10), torch.randn(10, 10))
            prof.step()

    with tarfile.open(tmp_path / "step_2" / "trace.tar.gz", "r:gz") as tar:
        trace = json.load(tar.extractfile("trace.json"))
    events = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    matmuls = [event for event in events if event["name"] == "aten::matmul"]

    assert matmuls
    assert any(event.get("cat") == "python_function" for event in events) == enabled
    assert all(("Input Dims" in event.get("args", {})) == enabled for event in matmuls)
