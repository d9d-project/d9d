import pytest
import torch
from d9d.core.dist_context import EXPERT_DOMAIN, DeviceMeshParameters
from d9d.module.block.moe import MoELayer, RouterReplayRecorder
from d9d.module.parallelism.api import parallelize_expert_parallel
from torch import nn

from d9d_test.modules.block.moe.batch import MOE_HIDDEN_SIZE, build_moe_inputs, materialize_moe_inputs
from d9d_test.modules.helper import copy_params_local_to_dist, torch_seed

_NUM_EXPERTS = 32
_TOP_K = 4
_INTERMEDIATE = 256


def _build_moe(dtype: torch.dtype) -> MoELayer:
    with torch_seed(42):
        moe = (
            MoELayer(
                hidden_dim=MOE_HIDDEN_SIZE,
                num_grouped_experts=_NUM_EXPERTS,
                intermediate_dim_grouped=_INTERMEDIATE,
                top_k=_TOP_K,
                router_renormalize_probabilities=True,
            )
            .cuda()
            .to(dtype)
        )
        moe.reset_parameters()
    return moe


class _StackedMoE(nn.Module):
    def __init__(self, num_layers: int, hidden_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MoELayer(
                    hidden_dim=hidden_dim,
                    intermediate_dim_grouped=2 * hidden_dim,
                    num_grouped_experts=num_experts,
                    top_k=top_k,
                    router_renormalize_probabilities=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states, replay_indices=None):
        for i, layer in enumerate(self.layers):
            layer_replay = replay_indices[:, i] if replay_indices is not None else None
            hidden_states = layer(hidden_states, replay_indices=layer_replay)
        return hidden_states

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


@pytest.mark.local
def test_install_enumerates_and_assigns_layer_ids():
    model = _StackedMoE(num_layers=3, hidden_dim=8, num_experts=4, top_k=2)
    recorder = RouterReplayRecorder()

    with recorder.install(model):
        for expected_id, layer in enumerate(model.layers):
            assert layer._replay_recorder is recorder
            assert layer._replay_layer_id == expected_id

    # context exit unbinds every layer
    for layer in model.layers:
        assert layer._replay_recorder is None
        assert layer._replay_layer_id is None


@pytest.mark.local
def test_tape_assembles_in_layer_order():
    model = _StackedMoE(num_layers=3, hidden_dim=8, num_experts=4, top_k=2)
    batch, seq, top_k = 2, 5, 2

    with RouterReplayRecorder().install(model) as rec:
        captures = [torch.randint(0, 4, (batch, seq, top_k)) for _ in model.layers]
        # capture out of order to prove tape() restores layer order
        for layer_id in reversed(range(len(captures))):
            rec.capture(layer_id, captures[layer_id])
        tape = rec.tape()

    assert tape.shape == (batch, 3, seq, top_k)
    for layer_id, indices in enumerate(captures):
        assert torch.equal(tape[:, layer_id], indices)


@pytest.mark.local
def test_tape_requires_full_recording():
    model = _StackedMoE(num_layers=2, hidden_dim=8, num_experts=4, top_k=2)

    with RouterReplayRecorder().install(model) as rec:
        with pytest.raises(RuntimeError, match="No routing was recorded"):
            rec.tape()

        rec.capture(0, torch.randint(0, 4, (1, 3, 2)))
        with pytest.raises(RuntimeError, match="Expected 2 layers to record"):
            rec.tape()


@pytest.mark.local
def test_double_install_is_rejected():
    model = _StackedMoE(num_layers=1, hidden_dim=8, num_experts=4, top_k=2)
    recorder = RouterReplayRecorder()

    with recorder.install(model), pytest.raises(RuntimeError, match="already installed"):  # noqa: SIM117
        with recorder.install(model):
            pass


@pytest.mark.local
def test_recorder_is_reusable_after_context_exit():
    model = _StackedMoE(num_layers=1, hidden_dim=8, num_experts=4, top_k=2)
    recorder = RouterReplayRecorder()

    with recorder.install(model) as rec:
        rec.capture(0, torch.randint(0, 4, (1, 2, 2)))
        rec.tape()

    # a second installation starts from a clean slate
    with recorder.install(model) as rec:
        rec.capture(0, torch.zeros(1, 2, 2, dtype=torch.long))
        assert torch.equal(rec.tape()[:, 0], torch.zeros(1, 2, 2, dtype=torch.long))


@pytest.mark.local
def test_record_then_replay_reproduces_routing():
    torch.manual_seed(7)
    model = _StackedMoE(num_layers=2, hidden_dim=64, num_experts=8, top_k=2).cuda().to(torch.bfloat16)
    model.reset_parameters()
    hidden_states = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)

    with RouterReplayRecorder().install(model) as rec:
        recorded_out = model(hidden_states)
        tape = rec.tape()

    replayed_out = model(hidden_states, replay_indices=tape)

    assert tape.shape == (2, 2, 16, 2)
    # replaying the recorded selection on identical weights/inputs must reproduce the forward exactly
    torch.testing.assert_close(replayed_out, recorded_out)


@pytest.mark.local
def test_replay_overrides_routing_and_changes_output():
    torch.manual_seed(11)
    model = _StackedMoE(num_layers=1, hidden_dim=64, num_experts=8, top_k=2).cuda().to(torch.bfloat16)
    model.reset_parameters()
    hidden_states = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)

    with RouterReplayRecorder().install(model) as rec:
        own_out = model(hidden_states)
        own_tape = rec.tape()

    # force a different (rotated) expert selection and confirm the output actually changes
    forced_tape = (own_tape + 1) % 8
    forced_out = model(hidden_states, replay_indices=forced_tape)

    assert not torch.allclose(forced_out, own_out)


@pytest.mark.distributed
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_recorded_routing_replays_under_expert_parallel(dtype: torch.dtype, dist_ctx_factory):
    ctx = dist_ctx_factory(DeviceMeshParameters(expert_parallel=8, data_parallel_replicate=8))
    init = build_moe_inputs(dtype)

    # Record the routing of the local (non-sharded) layer and confirm replay reproduces its forward exactly.
    local_inputs = materialize_moe_inputs(init)
    local = _build_moe(dtype)
    local_hidden = local_inputs.hidden_states + local_inputs.pre

    with RouterReplayRecorder().install(local) as recorder:
        out_local_record = local(local_hidden)
        tape = recorder.tape()

    assert tape.shape[1] == 1  # a single MoE layer
    replay = tape[:, 0]

    out_local_replay = local(local_hidden, replay_indices=replay)
    torch.testing.assert_close(out_local_replay, out_local_record, atol=2e-3, rtol=1e-2)

    # The expert-parallel sharded layer, given the same weights and the same replayed selection, must match.
    dist_inputs = materialize_moe_inputs(init)
    dist_model = _build_moe(dtype)
    parallelize_expert_parallel(
        dist_model,
        mesh_experts=ctx.mesh_for(EXPERT_DOMAIN)["ep_replicate", "ep_shard"],
    )
    copy_params_local_to_dist(local, dist_model)

    out_dist_replay = dist_model(dist_inputs.hidden_states + dist_inputs.pre, replay_indices=replay)
    torch.testing.assert_close(out_dist_replay, out_local_replay, atol=2e-3, rtol=1e-2)
