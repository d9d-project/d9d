import pytest
import torch
from d9d.module.block.moe import MoELayer, SigmoidGroupedTopKRouter
from torch import nn

from d9d_test.modules.block.moe.batch import MOE_HIDDEN_SIZE
from d9d_test.modules.helper import torch_seed

_DIM = 64
_NUM_EXPERTS = 16
_N_GROUP = 4  # 4 groups of 4 experts each
_TOPK_GROUP = 2  # select 2 groups
_TOP_K = 4  # select 4 experts total
_SCALING = 2.5
_HF_NUM_EXPERTS = 32
_HF_N_GROUP = 4
_HF_TOPK_GROUP = 2
_HF_TOP_K = 4
_HF_SCALING = 2.5


def _make_router(
    *,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = _SCALING,
    num_experts: int = _NUM_EXPERTS,
    n_group: int = _N_GROUP,
    topk_group: int = _TOPK_GROUP,
    top_k: int = _TOP_K,
) -> SigmoidGroupedTopKRouter:
    torch.manual_seed(0)
    router = SigmoidGroupedTopKRouter(
        dim=_DIM,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        norm_topk_prob=norm_topk_prob,
    )
    router.reset_parameters()
    return router


def test_output_shapes():
    router = _make_router()
    num_tokens = 32
    x = torch.randn(num_tokens, _DIM)
    result = router(x)
    assert result.selected_expert_indices.shape == (num_tokens, _TOP_K)
    assert result.selected_probabilities.shape == (num_tokens, _TOP_K)


def test_indices_dtype_is_int():
    router = _make_router()
    result = router(torch.randn(8, _DIM))
    assert result.selected_expert_indices.dtype in {torch.int32, torch.int64}


def test_weights_dtype_is_float32():
    router = _make_router()
    result = router(torch.randn(8, _DIM))
    assert result.selected_probabilities.dtype == torch.float32


def test_indices_in_valid_range():
    router = _make_router()
    result = router(torch.randn(64, _DIM))
    assert result.selected_expert_indices.min() >= 0
    assert result.selected_expert_indices.max() < _NUM_EXPERTS


def test_group_constraint():
    """Every token's selected experts must all fall within at most topk_group groups."""
    router = _make_router()
    result = router(torch.randn(128, _DIM))

    experts_per_group = _NUM_EXPERTS // _N_GROUP
    group_ids = result.selected_expert_indices // experts_per_group  # (T, top_k)
    for token_group_ids in group_ids:
        unique_groups = token_group_ids.unique()
        assert unique_groups.numel() <= _TOPK_GROUP, (
            f"Token routed to {unique_groups.numel()} groups, expected ≤ {_TOPK_GROUP}"
        )


def test_no_duplicate_experts_per_token():
    router = _make_router()
    result = router(torch.randn(64, _DIM))
    for row in result.selected_expert_indices:
        assert row.unique().numel() == _TOP_K, "Duplicate expert selected for a single token"


def test_norm_topk_prob_weights_sum_to_one():
    router = _make_router(norm_topk_prob=True, routed_scaling_factor=1.0)
    result = router(torch.randn(64, _DIM))
    sums = result.selected_probabilities.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=0)


def test_routed_scaling_factor_applied():
    """Weights with scale=k should be exactly k times the scale=1 weights."""
    x = torch.randn(32, _DIM)
    router1 = _make_router(norm_topk_prob=True, routed_scaling_factor=1.0)
    router2 = _make_router(norm_topk_prob=True, routed_scaling_factor=_SCALING)
    r1 = router1(x)
    r2 = router2(x)
    assert torch.equal(r1.selected_expert_indices, r2.selected_expert_indices)
    torch.testing.assert_close(r2.selected_probabilities, r1.selected_probabilities * _SCALING)


def test_correction_bias_affects_selection_not_weights():
    """
    e_score_correction_bias shifts which experts are chosen but must NOT change the
    weight values of the selected experts (weights come from unbiased sigmoid scores).
    """
    torch.manual_seed(7)
    x = torch.randn(64, _DIM)

    router_no_bias = _make_router(norm_topk_prob=False, routed_scaling_factor=1.0)
    router_biased = _make_router(norm_topk_prob=False, routed_scaling_factor=1.0)
    # Force expert 0 to always win selection in the biased router
    router_biased.e_score_correction_bias[0] = 1e6

    r_no = router_no_bias(x)
    r_bi = router_biased(x)

    assert (r_bi.selected_expert_indices == 0).any(), "Bias should have forced expert 0 into selection"

    # For positions where the same expert index appears in both results, the weight
    # must be identical (same unbiased sigmoid score, same gate weights).
    for t in range(x.shape[0]):
        no_idx = r_no.selected_expert_indices[t]
        bi_idx = r_bi.selected_expert_indices[t]
        for pos_bi, expert in enumerate(bi_idx):
            pos_no = (no_idx == expert).nonzero(as_tuple=True)[0]
            if pos_no.numel() > 0:
                torch.testing.assert_close(
                    r_bi.selected_probabilities[t, pos_bi],
                    r_no.selected_probabilities[t, pos_no[0]],
                    atol=1e-6,
                    rtol=0,
                    msg=f"Token {t}, expert {expert}: weight differs despite identical gate weights",
                )


def test_invalid_n_group_raises():
    with pytest.raises(ValueError, match="divisible"):
        SigmoidGroupedTopKRouter(
            dim=64,
            num_experts=15,
            top_k=4,
            n_group=4,
            topk_group=2,
            routed_scaling_factor=1.0,
            norm_topk_prob=True,
        )


def test_moe_layer_with_sigmoid_router():
    """MoELayer with SigmoidGroupedTopKRouter should produce valid output shapes."""
    hidden_dim = 64
    moe = MoELayer(
        hidden_dim=hidden_dim,
        num_grouped_experts=16,
        intermediate_dim_grouped=128,
        router=SigmoidGroupedTopKRouter(
            dim=hidden_dim,
            num_experts=16,
            top_k=4,
            n_group=4,
            topk_group=2,
            routed_scaling_factor=2.5,
            norm_topk_prob=True,
        ),
    )
    moe.reset_parameters()
    out = moe(torch.randn(8, 32, hidden_dim))
    assert out.shape == (8, 32, hidden_dim)


def _hf_route_reference(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    top_k: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Self-contained re-implementation of HF DSv3 routing logic for test reference."""
    num_experts = gate_weight.shape[0]
    experts_per_group = num_experts // n_group
    num_tokens = hidden_states.shape[0]

    scores = (hidden_states.float() @ gate_weight.float().T).sigmoid()
    scores_for_sel = scores + correction_bias

    group_scores = scores_for_sel.view(num_tokens, n_group, experts_per_group).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0)
    expert_mask = group_mask.unsqueeze(-1).expand(-1, -1, experts_per_group).reshape(num_tokens, num_experts).bool()
    masked = scores_for_sel.masked_fill(~expert_mask, float("-inf"))
    indices = torch.topk(masked, k=top_k, sorted=False)[1]
    weights = scores.gather(1, indices)
    if norm_topk_prob:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return indices, weights


@pytest.mark.local
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_consistent_with_hf_dsv3_routing_reference(dtype: torch.dtype):
    """
    d9d SigmoidGroupedTopKRouter must produce the same routing indices and expert weights
    as the HF DeepSeek-V3 routing reference when given identical gate weights and inputs.
    """
    with torch_seed(99):
        gate = nn.Linear(MOE_HIDDEN_SIZE, _HF_NUM_EXPERTS, bias=False).cuda().to(dtype)
        correction_bias = torch.zeros(_HF_NUM_EXPERTS, dtype=torch.float32, device="cuda")

        router = (
            SigmoidGroupedTopKRouter(
                dim=MOE_HIDDEN_SIZE,
                num_experts=_HF_NUM_EXPERTS,
                top_k=_HF_TOP_K,
                n_group=_HF_N_GROUP,
                topk_group=_HF_TOPK_GROUP,
                routed_scaling_factor=_HF_SCALING,
                norm_topk_prob=True,
            )
            .cuda()
            .to(dtype)
        )

        # Share exact gate weights
        router.gate.weight.data.copy_(gate.weight.data)
        router.e_score_correction_bias.copy_(correction_bias)

        x = torch.randn(_HF_NUM_EXPERTS * 4, MOE_HIDDEN_SIZE, device="cuda", dtype=dtype)

    ref_indices, ref_weights = _hf_route_reference(
        x,
        gate.weight,
        correction_bias,
        n_group=_HF_N_GROUP,
        topk_group=_HF_TOPK_GROUP,
        top_k=_HF_TOP_K,
        norm_topk_prob=True,
        routed_scaling_factor=_HF_SCALING,
    )
    d9d_result = router(x)

    # Sort by index before comparing (topk order may differ)
    ref_order = ref_indices.argsort(dim=-1)
    d9d_order = d9d_result.selected_expert_indices.argsort(dim=-1)

    ref_indices_sorted = ref_indices.gather(1, ref_order)
    d9d_indices_sorted = d9d_result.selected_expert_indices.gather(1, d9d_order)
    assert torch.equal(ref_indices_sorted, d9d_indices_sorted), "Routing indices differ from reference"

    ref_weights_sorted = ref_weights.gather(1, ref_order)
    d9d_weights_sorted = d9d_result.selected_probabilities.gather(1, d9d_order)
    torch.testing.assert_close(d9d_weights_sorted, ref_weights_sorted, atol=1e-5, rtol=1e-5)
