import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from d9d.core.dist_context import DeviceMeshParameters
from d9d.module.block.moe import TopKRouter


def _build_router(num_experts: int, top_k: int, renormalize: bool, enable_expert_bias: bool = False) -> TopKRouter:
    torch.manual_seed(0)
    router = TopKRouter(
        dim=16,
        num_experts=num_experts,
        top_k=top_k,
        renormalize_probabilities=renormalize,
        enable_expert_bias=enable_expert_bias,
    )
    router.reset_parameters()
    if enable_expert_bias:
        # give the bias non-trivial values so it would change selection if it were consulted
        with torch.no_grad():
            router.expert_bias.copy_(torch.randn(num_experts))
    return router


@pytest.mark.local
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("enable_expert_bias", [False, True])
def test_replay_with_own_indices_is_identity(renormalize, enable_expert_bias):
    # Replaying the router's *own* selection must reproduce the exact same indices and probabilities.
    router = _build_router(num_experts=8, top_k=3, renormalize=renormalize, enable_expert_bias=enable_expert_bias)
    hidden_states = torch.randn(10, 16)

    base = router(hidden_states)
    replayed = router(hidden_states, replay_indices=base.selected_expert_indices)

    assert torch.equal(replayed.selected_expert_indices, base.selected_expert_indices)
    torch.testing.assert_close(replayed.selected_probabilities, base.selected_probabilities)


@pytest.mark.local
def test_replay_forces_foreign_selection():
    # Replaying a *different* expert set must override the router's own top-k choice.
    router = _build_router(num_experts=8, top_k=2, renormalize=True)
    hidden_states = torch.randn(5, 16)

    own = router(hidden_states)
    # pick experts that are deliberately NOT the router's own top-k for each token
    forced = torch.empty_like(own.selected_expert_indices)
    for token in range(hidden_states.shape[0]):
        chosen = own.selected_expert_indices[token].tolist()
        alternatives = [e for e in range(8) if e not in chosen][:2]
        forced[token] = torch.tensor(alternatives)

    replayed = router(hidden_states, replay_indices=forced)
    assert torch.equal(replayed.selected_expert_indices, forced)


@pytest.mark.local
def test_replay_matches_r3_gating_formula():
    # The renormalized replay weights must equal R3 Eq. 8: softmax of the training logits restricted to the
    # selected experts, i.e. g_i = e^{s_i} / sum_{j in I} e^{s_j}.
    router = _build_router(num_experts=8, top_k=3, renormalize=True)
    hidden_states = torch.randn(7, 16)

    indices = torch.randint(0, 8, (7, 3))
    # de-duplicate per row so gather/softmax reference is well-defined
    for row in range(7):
        indices[row] = torch.randperm(8)[:3]

    result = router(hidden_states, replay_indices=indices)

    scores = router.gate(hidden_states)
    expected = F.softmax(scores.gather(dim=-1, index=indices), dim=-1, dtype=torch.float32)
    torch.testing.assert_close(result.selected_probabilities, expected)


@pytest.mark.local
def test_replay_gradient_flows_into_gate():
    # Selection is fixed, but the gate must still receive gradient through the replayed gating weights.
    router = _build_router(num_experts=8, top_k=2, renormalize=True)
    hidden_states = torch.randn(4, 16)
    indices = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])

    result = router(hidden_states, replay_indices=indices)
    result.selected_probabilities.sum().backward()

    assert router.gate.weight.grad is not None
    assert torch.any(router.gate.weight.grad != 0)


@pytest.mark.local
def test_replay_accepts_compact_integer_dtype():
    # Recorded indices may use a compact integer dtype; the router must cast for gather/downstream use.
    router = _build_router(num_experts=8, top_k=2, renormalize=True)
    hidden_states = torch.randn(3, 16)
    indices = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int16)

    result = router(hidden_states, replay_indices=indices)
    assert result.selected_expert_indices.dtype == torch.long


@pytest.mark.local
@pytest.mark.parametrize("bad_shape", [(3, 2), (4, 3), (4,)])
def test_replay_rejects_wrong_shape(bad_shape):
    router = _build_router(num_experts=8, top_k=2, renormalize=True)
    hidden_states = torch.randn(4, 16)
    bad = torch.zeros(bad_shape, dtype=torch.long)

    with pytest.raises(ValueError, match="replay_indices must have shape"):
        router(hidden_states, replay_indices=bad)


@pytest.mark.distributed
@pytest.mark.parametrize("renormalize", [True, False])
def test_replay_is_consistent_across_ranks(renormalize, dist_ctx_factory):
    # The router gate is replicated across the cluster, so with identical inputs every rank routes identically.
    # This is what makes a routing tape recorded on one rank replayable on all of them. We verify both that replay
    # reproduces the recorded selection/probabilities locally and that every rank agreed on the same selection.
    dist_ctx_factory(DeviceMeshParameters(expert_parallel=8, data_parallel_replicate=8))

    router = _build_router(num_experts=32, top_k=4, renormalize=renormalize).cuda()
    hidden_states = torch.randn(64, 16, device="cuda")

    base = router(hidden_states)
    replayed = router(hidden_states, replay_indices=base.selected_expert_indices)

    assert torch.equal(replayed.selected_expert_indices, base.selected_expert_indices)
    torch.testing.assert_close(replayed.selected_probabilities, base.selected_probabilities)

    # all ranks must have selected exactly the same experts (max == min == local everywhere)
    selection = base.selected_expert_indices.to(torch.float64)
    selection_max = selection.clone()
    selection_min = selection.clone()
    dist.all_reduce(selection_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(selection_min, op=dist.ReduceOp.MIN)
    assert torch.equal(selection_max, selection)
    assert torch.equal(selection_min, selection)
