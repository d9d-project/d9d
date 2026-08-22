import re

import pytest
import torch
from d9d.module.block.moe import MoELayer
from d9d.module.block.moe.layer import MoELayer as _MoELayer
from torch import nn

from d9d_test.modules.helper.routing_replay import d9d_routing_replay, zero_hf_router_gradients


class _LayerHolder(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp


class _FakeBackbone(nn.Module):
    def __init__(self, layers: nn.ModuleDict):
        super().__init__()
        self.layers = layers


class _FakeD9DModel(nn.Module):
    def __init__(self, mlps: list[nn.Module]):
        super().__init__()
        self.model = _FakeBackbone(nn.ModuleDict({str(i): _LayerHolder(m) for i, m in enumerate(mlps)}))


def _make_moe_layer(hidden_dim: int = 8, num_experts: int = 4, top_k: int = 2) -> MoELayer:
    """Constructs a small MoELayer. CPU-safe to construct; forward needs GPU."""
    return MoELayer(
        hidden_dim=hidden_dim,
        intermediate_dim_grouped=hidden_dim * 2,
        num_grouped_experts=num_experts,
        top_k=top_k,
        router_renormalize_probabilities=True,
    )


def _make_hf_shaped_module(prefix: str) -> nn.Module:
    """Builds a module hierarchy that produces parameter names at:
        {prefix}layers.{i}.mlp.gate.{weight,bias}
        {prefix}layers.{i}.mlp.shared_experts.gate_proj.weight
        {prefix}layers.{i}.self_attn.q_proj.weight
        {prefix}embed_tokens.weight
    where ``prefix`` is either "model." or "".
    """

    class Gate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(3))
            self.bias = nn.Parameter(torch.ones(3))

    class SharedExperts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(3, 3, bias=False)

    class Mlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = Gate()
            self.shared_experts = SharedExperts()

    class SelfAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(3, 3, bias=False)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = Mlp()
            self.self_attn = SelfAttn()

    class Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(10, 3)
            self.layers = nn.ModuleList([Layer(), Layer()])

    if prefix == "model.":

        class Wrapped(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = Backbone()

        return Wrapped()
    return Backbone()


@pytest.mark.parametrize("prefix", ["model.", ""])
def test_zero_hf_router_gradients_only_touches_router_gate(prefix: str) -> None:
    model = _make_hf_shaped_module(prefix)
    for p in model.parameters():
        p.grad = torch.ones_like(p)

    zero_hf_router_gradients(model)

    expected_zero = re.compile(rf"^{re.escape(prefix)}layers\.\d+\.mlp\.gate\.weight$")
    for name, p in model.named_parameters():
        assert p.grad is not None
        if expected_zero.match(name):
            assert torch.equal(p.grad, torch.zeros_like(p)), f"{name} should be zeroed"
        else:
            assert torch.equal(p.grad, torch.ones_like(p)), f"{name} should not be zeroed"


def test_zero_hf_router_gradients_skips_params_without_grad() -> None:
    """Parameters with ``grad is None`` are left alone (no AttributeError)."""
    model = _make_hf_shaped_module(prefix="model.")
    zero_hf_router_gradients(model)
    for _, p in model.named_parameters():
        assert p.grad is None


def test_replay_swaps_and_restores_forward_on_normal_exit() -> None:
    mlp = _make_moe_layer()
    model = _FakeD9DModel([mlp])
    original_forward = _MoELayer.forward

    with d9d_routing_replay(model, routing={}, batch_size=1):
        assert _MoELayer.forward is not original_forward

    assert _MoELayer.forward is original_forward


def test_replay_restores_forward_on_exception() -> None:
    """Simulate the test body raising mid-block (e.g. an assertion failure or a
    backward-pass error) and verify the monkey-patch is still rolled back, so
    later tests do not inherit the patched ``MoELayer.forward``."""
    mlp = _make_moe_layer()
    model = _FakeD9DModel([mlp])
    original_forward = _MoELayer.forward

    sentinel = "simulated test failure inside replay block"
    with pytest.raises(RuntimeError, match=sentinel), d9d_routing_replay(model, routing={}, batch_size=1):
        raise RuntimeError(sentinel)

    assert _MoELayer.forward is original_forward


def test_repeated_enter_exit_does_not_leak_patch() -> None:
    """Running the context manager multiple times in sequence leaves no residue
    — important because pytest can run replay-using tests back to back."""
    mlp = _make_moe_layer()
    model = _FakeD9DModel([mlp])
    original_forward = _MoELayer.forward

    for _ in range(3):
        with d9d_routing_replay(model, routing={}, batch_size=1):
            pass
        assert _MoELayer.forward is original_forward


def _has_cuda() -> bool:
    return torch.cuda.is_available()


@pytest.mark.local
@pytest.mark.skipif(not _has_cuda(), reason="MoE kernels require CUDA")
def test_replay_overrides_native_routing() -> None:
    """End-to-end: with a routing dict that sends every token to experts {0, 1}
    only, the layer's ``tokens_per_expert`` after a single forward reflects
    that — not whatever the native router (initialized random) would have done.
    """
    torch.manual_seed(0)
    hidden_dim, num_experts, top_k = 8, 4, 2
    batch_size, seq_len = 2, 4
    num_tokens = batch_size * seq_len

    mlp = _make_moe_layer(hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k).cuda().bfloat16()
    mlp.reset_parameters()
    model = _FakeD9DModel([mlp])

    # Inject routing: every (batch, position) routes to experts (0, 1) with equal weight.
    full_indices = torch.zeros((num_tokens, top_k), dtype=torch.long, device="cuda")
    full_indices[:, 1] = 1
    full_weights = torch.full((num_tokens, top_k), 0.5, dtype=torch.bfloat16, device="cuda")
    routing = {0: (full_indices, full_weights)}

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16, device="cuda")

    mlp.reset_stats()
    with d9d_routing_replay(model, routing, batch_size=batch_size):
        _ = mlp(hidden_states)

    tpe = mlp.tokens_per_expert.cpu().tolist()
    assert tpe[0] == num_tokens, f"expert 0 should have {num_tokens} tokens, got {tpe[0]}"
    assert tpe[1] == num_tokens, f"expert 1 should have {num_tokens} tokens, got {tpe[1]}"
    assert tpe[2] == 0, f"expert 2 should have 0 tokens, got {tpe[2]}"
    assert tpe[3] == 0, f"expert 3 should have 0 tokens, got {tpe[3]}"


@pytest.mark.local
@pytest.mark.skipif(not _has_cuda(), reason="MoE kernels require CUDA")
def test_replay_falls_through_for_layers_not_in_routing() -> None:
    """Layers whose index is absent from the routing dict run through the
    native router, not the replay closure."""
    torch.manual_seed(0)
    hidden_dim, num_experts, top_k = 8, 4, 2
    batch_size, seq_len = 2, 4

    mlp_a = _make_moe_layer(hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k).cuda().bfloat16()
    mlp_b = _make_moe_layer(hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k).cuda().bfloat16()
    mlp_a.reset_parameters()
    mlp_b.reset_parameters()
    model = _FakeD9DModel([mlp_a, mlp_b])

    num_tokens = batch_size * seq_len
    forced_indices = torch.full((num_tokens, top_k), 3, dtype=torch.long, device="cuda")
    forced_weights = torch.full((num_tokens, top_k), 0.5, dtype=torch.bfloat16, device="cuda")
    routing = {0: (forced_indices, forced_weights)}

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16, device="cuda")

    mlp_a.reset_stats()
    mlp_b.reset_stats()
    with d9d_routing_replay(model, routing, batch_size=batch_size):
        out_a = mlp_a(hidden_states)
        _ = mlp_b(out_a)

    # Layer 0: every (token, slot) pair routed to expert 3 → tpe[3] == top_k * num_tokens.
    tpe_a = mlp_a.tokens_per_expert.cpu().tolist()
    assert tpe_a[3] == top_k * num_tokens
    assert sum(tpe_a[:3]) == 0

    # Layer 1: total count should equal top_k * num_tokens, but distribution is the
    # native router's choice (not all on expert 3 with overwhelming probability —
    # uniform-init router will spread roughly evenly).
    tpe_b = mlp_b.tokens_per_expert.cpu().tolist()
    assert sum(tpe_b) == top_k * num_tokens
    # Native routing will not place 100% on expert 3 with random weights.
    assert tpe_b[3] < top_k * num_tokens


@pytest.mark.local
@pytest.mark.skipif(not _has_cuda(), reason="MoE kernels require CUDA")
def test_replay_trims_captured_routing_to_d9d_sequence_length() -> None:
    """When HF was run on S_hf and d9d on S_d9d < S_hf, the captured routing
    is sliced to (B, S_d9d, top_k). The first S_d9d positions of HF's routing
    govern the corresponding d9d tokens; HF positions [S_d9d:] are discarded."""
    torch.manual_seed(0)
    hidden_dim, num_experts, top_k = 8, 4, 2
    batch_size, s_hf, s_d9d = 2, 6, 5
    num_tokens_hf = batch_size * s_hf

    mlp = _make_moe_layer(hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k).cuda().bfloat16()
    mlp.reset_parameters()
    model = _FakeD9DModel([mlp])

    # HF routing: first S_d9d positions in each batch row → expert 0; trailing position → expert 3.
    # If the trim is correct, no token goes to expert 3 during the d9d forward.
    full_indices = torch.zeros((num_tokens_hf, top_k), dtype=torch.long, device="cuda")
    full_indices[:, 1] = 1
    # Mark the trailing positions (one per batch row) as expert 3.
    full_indices_view = full_indices.reshape(batch_size, s_hf, top_k)
    full_indices_view[:, s_d9d:, :] = 3
    full_weights = torch.full((num_tokens_hf, top_k), 0.5, dtype=torch.bfloat16, device="cuda")
    routing = {0: (full_indices, full_weights)}

    hidden_states = torch.randn(batch_size, s_d9d, hidden_dim, dtype=torch.bfloat16, device="cuda")

    mlp.reset_stats()
    with d9d_routing_replay(model, routing, batch_size=batch_size):
        _ = mlp(hidden_states)

    tpe = mlp.tokens_per_expert.cpu().tolist()
    expected_per_kept_expert = batch_size * s_d9d
    assert tpe[0] == expected_per_kept_expert, f"expert 0: {tpe[0]} != {expected_per_kept_expert}"
    assert tpe[1] == expected_per_kept_expert, f"expert 1: {tpe[1]} != {expected_per_kept_expert}"
    assert tpe[3] == 0, f"expert 3 should be discarded by the trim, got {tpe[3]}"
