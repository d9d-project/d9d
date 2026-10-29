import pytest
import torch
from d9d.loop.component import PipelineStateHandler
from torch.testing import assert_close


@pytest.mark.local
def test_store_and_read_same_microbatch():
    handler = PipelineStateHandler()

    handler.store(0, {"labels": torch.tensor([1, 2, 3])})

    with handler.scope(0) as state:
        assert_close(state["labels"], torch.tensor([1, 2, 3]))


@pytest.mark.local
def test_microbatches_are_isolated():
    handler = PipelineStateHandler()

    handler.store(0, {"loss": torch.tensor(0.5)})
    handler.store(1, {"loss": torch.tensor(1.5)})

    with handler.scope(0) as state_0:
        assert_close(state_0["loss"], torch.tensor(0.5))
    with handler.scope(1) as state_1:
        assert_close(state_1["loss"], torch.tensor(1.5))


@pytest.mark.local
def test_reset_clears_all_state():
    handler = PipelineStateHandler()

    handler.store(0, {"x": torch.tensor([1.0])})
    handler.reset()

    with pytest.raises(KeyError), handler.scope(0):
        pass


@pytest.mark.local
def test_store_detaches_tensor():
    handler = PipelineStateHandler()

    t_grad = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    handler.store(0, {"grad_tensor": t_grad})

    with handler.scope(0) as state:
        stored = state["grad_tensor"]
    assert stored.requires_grad is False
    assert stored.grad_fn is None
    assert_close(stored, t_grad.detach())


@pytest.mark.local
def test_store_detaches_inside_pytree():
    handler = PipelineStateHandler()

    payload = {"a": torch.tensor([1.0], requires_grad=True), "b": [torch.tensor([2.0], requires_grad=True)]}
    handler.store(0, payload)

    with handler.scope(0) as state:
        assert state["a"].requires_grad is False
        assert state["b"][0].requires_grad is False


@pytest.mark.local
def test_non_tensor_values_pass_through():
    handler = PipelineStateHandler()

    handler.store(0, {"ids": [10, 20, 30], "count": 7})

    with handler.scope(0) as state:
        assert state["ids"] == [10, 20, 30]
        assert state["count"] == 7


@pytest.mark.local
def test_scope_detaches_values_written_inside():
    handler = PipelineStateHandler()

    handler.store(0, {})

    # Emulate stashing a graph-attached model output during loss computation.
    graph_tensor = (torch.tensor([1.0, 2.0], requires_grad=True) * 2).sum()
    with handler.scope(0) as state:
        state["cached"] = graph_tensor
        # Still attached while the scope is open.
        assert state["cached"].requires_grad is True

    # Detached once the scope exits, so the cached graph is not kept alive.
    with handler.scope(0) as state:
        assert state["cached"].requires_grad is False
        assert state["cached"].grad_fn is None
