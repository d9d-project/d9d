import pytest
import torch
import torch.distributed as dist
from d9d.core import dist_ops
from d9d.core.dist_context import REGULAR_DOMAIN


@pytest.fixture(scope="session")
def group(dist_ctx_dpr) -> dist.ProcessGroup:
    return dist_ctx_dpr.mesh_for(REGULAR_DOMAIN).get_group("dp_replicate")


def test_all_gather_object(group):
    rank = group.rank()
    size = group.size()

    send_obj = {"rank": rank, "val": rank * 10}

    result = dist_ops.all_gather_object(send_obj, group)

    assert len(result) == size
    for i, item in enumerate(result):
        assert item == {"rank": i, "val": i * 10}


@pytest.mark.distributed
@pytest.mark.parametrize("send_to", [0, 7])
def test_gather_object(group, send_to: int):
    rank = group.rank()
    size = group.size()

    send_obj = f"data_from_{rank}"

    res = dist_ops.gather_object(send_obj, group, group_dst=send_to)

    if rank == send_to:
        assert isinstance(res, list)
        assert len(res) == size
        expected = [f"data_from_{i}" for i in range(size)]
        assert res == expected
    else:
        assert res is None


@pytest.mark.distributed
def test_all_gather_tensor_sync(group):
    rank = group.rank()

    # Tensor: [rank, rank]
    t = torch.tensor([rank, rank], dtype=torch.float32, device="cuda")

    result = dist_ops.all_gather(t, group, async_op=False)

    assert isinstance(result, list)
    assert len(result) == group.size()

    for i, tensor in enumerate(result):
        expected = torch.tensor([i, i], dtype=torch.float32, device="cuda")
        assert torch.equal(tensor, expected)


@pytest.mark.distributed
def test_all_gather_tensor_async(group):
    rank = group.rank()

    t = torch.tensor([float(rank)], device="cuda")

    result, work = dist_ops.all_gather(t, group, async_op=True)

    assert isinstance(work, dist.Work)
    work.wait()

    for i, tensor in enumerate(result):
        assert tensor.item() == float(i)


@pytest.mark.distributed
@pytest.mark.parametrize("send_to", [0, 7])
def test_gather_tensor(group, send_to: int):
    rank = group.rank()
    size = group.size()

    t = torch.ones(2, device="cuda") * rank

    res = dist_ops.gather(t, group, group_dst=send_to, async_op=False)

    if rank == send_to:
        assert len(res) == size
        for i, tensor in enumerate(res):
            assert torch.equal(tensor, torch.ones(2, device="cuda") * i)
    else:
        assert res is None


@pytest.mark.distributed
@pytest.mark.parametrize("send_to", [0, 7])
def test_gather_tensor_async(group, send_to: int):
    rank = group.rank()
    size = group.size()

    t = torch.ones(2, device="cuda") * rank

    res, work = dist_ops.gather(t, group, group_dst=send_to, async_op=True)
    assert isinstance(work, dist.Work)
    work.wait()

    if rank == send_to:
        assert len(res) == size
        for i, tensor in enumerate(res):
            assert torch.equal(tensor, torch.ones(2, device="cuda") * i)
    else:
        assert res is None


@pytest.mark.distributed
def test_all_gather_variadic_shape(group):
    rank = group.rank()
    size = group.size()

    # Create variadic shapes
    # Rank 0: (1, 2), Rank 1: (2, 2), etc.
    t = torch.ones((rank + 1, 2), device="cuda") * rank

    result = dist_ops.all_gather_variadic_shape(t, group, async_op=False)

    assert len(result) == size
    for i, tensor in enumerate(result):
        expected_shape = (i + 1, 2)
        assert tensor.shape == expected_shape
        assert torch.all(tensor == i)


@pytest.mark.distributed
def test_all_gather_variadic_shape_async(group):
    rank = group.rank()
    size = group.size()

    t = torch.ones((rank + 1, 2), device="cuda") * rank

    result, work = dist_ops.all_gather_variadic_shape(t, group, async_op=True)

    assert isinstance(work, dist.Work)
    work.wait()

    assert len(result) == size
    for i, tensor in enumerate(result):
        expected_shape = (i + 1, 2)
        assert tensor.shape == expected_shape
        assert torch.all(tensor == i)


@pytest.mark.distributed
@pytest.mark.parametrize("send_to", [0, 7])
def test_gather_variadic_shape(group, send_to: int):
    rank = group.rank()
    size = group.size()

    t = torch.zeros(rank + 1, device="cuda")

    res = dist_ops.gather_variadic_shape(t, group, group_dst=send_to)

    if rank == send_to:
        assert len(res) == size
        for i, tensor in enumerate(res):
            assert tensor.shape == (i + 1,)
    else:
        assert res is None
