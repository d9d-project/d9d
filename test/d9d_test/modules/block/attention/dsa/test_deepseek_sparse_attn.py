import dataclasses

import pytest
import torch
from d9d.core.dist_context import BATCH_DOMAIN, DENSE_DOMAIN, DeviceMeshParameters
from d9d.module.block.attention import DeepSeekSparseAttention, GroupedQueryAttention
from d9d.module.block.attention.sdpa import EagerSdpaBackendConfig
from d9d.module.block.positional import RotaryEmbeddingStyle
from d9d.module.block.positional.rope import prepare_rotary_cos_sin_emb
from d9d.module.parallelism.api import parallelize_hsdp
from torch import nn
from torch.testing import assert_close

from d9d_test.modules.helper import (
    check_grad_distance_all_local_dist,
    copy_params_local_to_dist,
    sync_grads_manually,
    torch_seed,
)

_HIDDEN = 64
_HEADS = 8
_HEAD_DIM = 8
_INDEX_HEADS = 4
_INDEX_HEAD_DIM = 16
_BATCH = 2
_SEQ = 16

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.float32


def _position_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = prepare_rotary_cos_sin_emb(
        rope_base=10000,
        head_dim=_HEAD_DIM,
        max_position_ids=_SEQ,
        device=torch.device(_DEVICE),
        dtype=_DTYPE,
        style=RotaryEmbeddingStyle.HALF,
    )
    return cos.unsqueeze(0).expand(_BATCH, -1, -1), sin.unsqueeze(0).expand(_BATCH, -1, -1)


def _build_dsa(top_k: int, num_key_value_heads: int = _HEADS) -> DeepSeekSparseAttention:
    torch.manual_seed(42)
    module = DeepSeekSparseAttention(
        hidden_size=_HIDDEN,
        num_attention_heads=_HEADS,
        num_key_value_heads=num_key_value_heads,
        head_dim=_HEAD_DIM,
        index_n_heads=_INDEX_HEADS,
        index_head_dim=_INDEX_HEAD_DIM,
        index_top_k=top_k,
        qk_norm_eps=None,
        rope_style=RotaryEmbeddingStyle.HALF,
        sdpa_backend=EagerSdpaBackendConfig(),
    ).to(device=_DEVICE, dtype=_DTYPE)
    module.reset_parameters()
    return module


@pytest.mark.local
@pytest.mark.parametrize("num_key_value_heads", [_HEADS, 2, 1])
def test_output_shape(num_key_value_heads: int) -> None:
    module = _build_dsa(top_k=5, num_key_value_heads=num_key_value_heads)
    x = torch.randn(_BATCH, _SEQ, _HIDDEN, device=_DEVICE, dtype=_DTYPE)
    out = module(x, attention_mask=None, position_embeddings=_position_embeddings())
    assert out.shape == (_BATCH, _SEQ, _HIDDEN)


@pytest.mark.local
def test_attended_count_never_exceeds_top_k() -> None:
    """Each query attends to at most ``top_k`` causal tokens via the selection mask."""
    top_k = 5
    module = _build_dsa(top_k=top_k)
    x = torch.randn(_BATCH, _SEQ, _HIDDEN, device=_DEVICE, dtype=_DTYPE)

    positions = torch.arange(_SEQ, device=_DEVICE)
    causal_bias = torch.zeros(_SEQ, _SEQ, device=_DEVICE, dtype=_DTYPE).masked_fill_(
        positions.unsqueeze(0) > positions.unsqueeze(1), float("-inf")
    )
    mask = module.indexer(x, attention_bias=causal_bias) + causal_bias
    counts = (mask > float("-inf")).sum(dim=-1)
    assert int(counts.max()) <= top_k


@pytest.mark.local
def test_causal_property() -> None:
    """Outputs at positions < p must not depend on tokens at positions >= p."""
    module = _build_dsa(top_k=5)
    x = torch.randn(_BATCH, _SEQ, _HIDDEN, device=_DEVICE, dtype=_DTYPE)
    rope = _position_embeddings()

    out_full = module(x, attention_mask=None, position_embeddings=rope)

    cut = 6
    x_noisy = x.clone()
    x_noisy[:, cut:, :] = torch.randn(_BATCH, _SEQ - cut, _HIDDEN, device=_DEVICE, dtype=_DTYPE)
    out_noisy = module(x_noisy, attention_mask=None, position_embeddings=rope)

    assert_close(out_noisy[:, :cut], out_full[:, :cut], rtol=0, atol=1e-5)


@pytest.mark.local
def test_top_k_full_equals_dense_attention() -> None:
    """With ``top_k >= seq_len`` the selection is a no-op and DSA equals dense causal attention."""
    module = _build_dsa(top_k=_SEQ + 4)
    x = torch.randn(_BATCH, _SEQ, _HIDDEN, device=_DEVICE, dtype=_DTYPE)
    rope = _position_embeddings()

    reference = GroupedQueryAttention(
        hidden_size=_HIDDEN,
        num_attention_heads=_HEADS,
        num_key_value_heads=_HEADS,
        head_dim=_HEAD_DIM,
        qk_norm_eps=None,
        is_causal=True,
        rope_style=RotaryEmbeddingStyle.HALF,
        sdpa_backend=EagerSdpaBackendConfig(),
    ).to(device=_DEVICE, dtype=_DTYPE)
    reference.load_state_dict(module.attention.state_dict())

    out_dsa = module(x, attention_mask=None, position_embeddings=rope)
    out_dense = reference(x, attention_mask=None, position_embeddings=rope)
    assert_close(out_dsa, out_dense, rtol=0, atol=1e-5)


@pytest.mark.local
def test_main_loss_trains_attention_but_not_indexer() -> None:
    """The hard top-k selection is non-differentiable, so the main loss trains the attention
    projections (and the input) but leaves the indexer untouched, exactly as DeepSeek-V3.2
    optimizes the indexer separately through its auxiliary objective."""
    module = _build_dsa(top_k=5)
    x = torch.randn(_BATCH, _SEQ, _HIDDEN, device=_DEVICE, dtype=_DTYPE, requires_grad=True)

    out = module(x, attention_mask=None, position_embeddings=_position_embeddings())
    out.mean().backward()

    assert x.grad is not None
    assert module.attention.q_proj.weight.grad is not None
    assert module.indexer.weights_proj.weight.grad is None


@pytest.mark.local
def test_index_scores_are_differentiable() -> None:
    """The indexer is trainable through its (continuous) scores, which an auxiliary
    KL objective uses to align the indexer with the main attention distribution."""
    module = _build_dsa(top_k=5)
    x = torch.randn(_BATCH, _SEQ, _HIDDEN, device=_DEVICE, dtype=_DTYPE)

    module.indexer.index_scores(x).pow(2).mean().backward()
    assert module.indexer.weights_proj.weight.grad is not None
    assert module.indexer.q_proj.weight.grad is not None


def _manual_dsa_reference(
    module: DeepSeekSparseAttention, x: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Independent DSA forward: indexer top-k selection + masked grouped-query attention."""
    attn = module.attention
    b, s, _ = x.shape
    cos, sin = position_embeddings

    positions = torch.arange(s, device=x.device)
    causal = torch.zeros(s, s, device=x.device, dtype=x.dtype).masked_fill_(
        positions.unsqueeze(0) > positions.unsqueeze(1), float("-inf")
    )
    scores = module.indexer.index_scores(x, attention_bias=causal)
    top_k = min(module.indexer.top_k, s)
    selected = scores.topk(top_k, dim=-1).indices
    selection = torch.full((b, s, s), float("-inf"), device=x.device, dtype=x.dtype).scatter_(-1, selected, 0.0)
    mask = (selection + causal).unsqueeze(1)

    q = attn.q_proj(x).view(b, s, _HEADS, _HEAD_DIM)
    k = attn.k_proj(x).view(b, s, -1, _HEAD_DIM)
    v = attn.v_proj(x).view(b, s, -1, _HEAD_DIM)
    q, k = attn._apply_rope(q, k, cos, sin)

    groups = q.shape[2] // k.shape[2]
    q_h = q.transpose(1, 2)
    k_h = k.transpose(1, 2).repeat_interleave(groups, dim=1)
    v_h = v.transpose(1, 2).repeat_interleave(groups, dim=1)

    logits = (q_h @ k_h.transpose(-1, -2)) * (_HEAD_DIM**-0.5) + mask
    weights = torch.softmax(logits, dim=-1)
    out = (weights @ v_h).transpose(1, 2).reshape(b, s, _HEADS * _HEAD_DIM)
    return attn.o_proj(out)


@pytest.mark.local
@pytest.mark.parametrize("num_key_value_heads", [_HEADS, 2])
def test_matches_independent_masked_attention(num_key_value_heads: int) -> None:
    """The full sparse forward must equal an independent indexer-selection + masked GQA reference."""
    top_k = 5
    module = _build_dsa(top_k=top_k, num_key_value_heads=num_key_value_heads)
    x = torch.randn(_BATCH, _SEQ, _HIDDEN, device=_DEVICE, dtype=_DTYPE)
    rope = _position_embeddings()

    out_module = module(x, attention_mask=None, position_embeddings=rope)
    out_reference = _manual_dsa_reference(module, x, rope)

    # Sanity: the selection is genuinely sparse, not accidentally dense.
    assert top_k < _SEQ
    assert_close(out_module, out_reference, rtol=1e-5, atol=1e-5)


@pytest.mark.local
def test_default_backend_matches_eager() -> None:
    """The default (PyTorch SDPA) backend must agree with the explicit eager backend, confirming
    the folded causal + additive selection mask is handled correctly by a non-causal SDPA call."""
    x = torch.randn(_BATCH, _SEQ, _HIDDEN, device=_DEVICE, dtype=_DTYPE)
    rope = _position_embeddings()

    eager = _build_dsa(top_k=5)
    torch.manual_seed(42)
    default = DeepSeekSparseAttention(
        hidden_size=_HIDDEN,
        num_attention_heads=_HEADS,
        num_key_value_heads=_HEADS,
        head_dim=_HEAD_DIM,
        index_n_heads=_INDEX_HEADS,
        index_head_dim=_INDEX_HEAD_DIM,
        index_top_k=5,
        qk_norm_eps=None,
        rope_style=RotaryEmbeddingStyle.HALF,
    ).to(device=_DEVICE, dtype=_DTYPE)
    default.load_state_dict(eager.state_dict())

    out_eager = eager(x, attention_mask=None, position_embeddings=rope)
    out_default = default(x, attention_mask=None, position_embeddings=rope)
    assert_close(out_default, out_eager, rtol=1e-4, atol=1e-5)


_DIST_HIDDEN = 512
_DIST_HEADS = 8
_DIST_HEAD_DIM = 64
_DIST_INDEX_HEADS = 8
_DIST_INDEX_HEAD_DIM = 64
_DIST_TOP_K = 16
_DIST_BATCH = 2
_DIST_SEQ = 64
_DIST_NORM_EPS = 1e-6


@dataclasses.dataclass
class _DistInputsInit:
    hidden_states: torch.Tensor
    rope: tuple[torch.Tensor, torch.Tensor]
    pre_init: torch.Tensor


@dataclasses.dataclass
class _DistInputs:
    hidden_states: torch.Tensor
    rope: tuple[torch.Tensor, torch.Tensor]
    pre: torch.nn.Parameter


def _build_dist_inputs(dtype: torch.dtype) -> _DistInputsInit:
    with torch_seed(4242):
        hidden_states = torch.randn(_DIST_BATCH, _DIST_SEQ, _DIST_HIDDEN, device="cuda", dtype=dtype)
        cos, sin = prepare_rotary_cos_sin_emb(
            rope_base=10000,
            head_dim=_DIST_HEAD_DIM,
            max_position_ids=_DIST_SEQ,
            device=torch.device("cuda"),
            dtype=dtype,
            style=RotaryEmbeddingStyle.HALF,
        )
        return _DistInputsInit(
            hidden_states=hidden_states,
            rope=(cos.unsqueeze(0).expand(_DIST_BATCH, -1, -1), sin.unsqueeze(0).expand(_DIST_BATCH, -1, -1)),
            pre_init=torch.zeros((1, 1, _DIST_HIDDEN), device="cuda", dtype=dtype),
        )


def _materialize_dist_inputs(init: _DistInputsInit) -> _DistInputs:
    cos, sin = init.rope
    return _DistInputs(
        hidden_states=init.hidden_states.clone(),
        rope=(cos.clone(), sin.clone()),
        pre=torch.nn.Parameter(init.pre_init.clone()),
    )


def _build_dist_dsa(dtype: torch.dtype, num_key_value_heads: int) -> DeepSeekSparseAttention:
    with torch_seed(42):
        module = (
            DeepSeekSparseAttention(
                hidden_size=_DIST_HIDDEN,
                num_attention_heads=_DIST_HEADS,
                num_key_value_heads=num_key_value_heads,
                head_dim=_DIST_HEAD_DIM,
                index_n_heads=_DIST_INDEX_HEADS,
                index_head_dim=_DIST_INDEX_HEAD_DIM,
                index_top_k=_DIST_TOP_K,
                qk_norm_eps=_DIST_NORM_EPS,
                rope_style=RotaryEmbeddingStyle.HALF,
            )
            .cuda()
            .to(dtype)
        )
        module.reset_parameters()
    return module


class _DsaWithAux(nn.Module):
    """Exposes the indexer scores alongside the attention output.

    DSA's hard top-k selection is non-differentiable, so the main loss never trains the
    indexer. This wrapper returns the (continuous) index scores as well, standing in for
    the indexer's auxiliary objective. Computing them inside this module's forward keeps
    the indexer parameters in the autograd graph and in the FSDP unshard window, so their
    gradients are reduced correctly and can be compared against the local reference.
    """

    def __init__(self, attention: DeepSeekSparseAttention) -> None:
        super().__init__()
        self.dsa = attention

    def forward(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.dsa(hidden_states, attention_mask=None, position_embeddings=position_embeddings)
        return out, self.dsa.indexer.index_scores(hidden_states)


def _dist_forward_and_loss(module: _DsaWithAux, inputs: _DistInputs) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = inputs.hidden_states + inputs.pre
    out, index_scores = module(hidden_states, inputs.rope)
    return out, out.sum() / _DIST_HIDDEN + index_scores.pow(2).mean()


@pytest.mark.distributed
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_key_value_heads", [_DIST_HEADS, 2])
@pytest.mark.parametrize(
    "mesh",
    [
        pytest.param(DeviceMeshParameters(data_parallel_shard=8), id="fsdp_8"),
        pytest.param(DeviceMeshParameters(data_parallel_replicate=2, data_parallel_shard=4), id="hsdp_2x4"),
    ],
)
def test_consistent_to_itself_hsdp(
    mesh: DeviceMeshParameters, dtype: torch.dtype, num_key_value_heads: int, dist_ctx_factory
) -> None:
    """Sharded (FSDP/HSDP) DSA must match a replicated local reference in both output and gradients."""
    ctx = dist_ctx_factory(mesh)
    init = _build_dist_inputs(dtype)

    # Local (replicated) reference.
    local_inputs = _materialize_dist_inputs(init)
    local = _DsaWithAux(_build_dist_dsa(dtype, num_key_value_heads))
    out_local, loss_local = _dist_forward_and_loss(local, local_inputs)
    loss_local.backward()

    # Sharded module over the dense mesh (HSDP folds to plain FSDP when replicate == 1).
    dist_inputs = _materialize_dist_inputs(init)
    dist_model = _DsaWithAux(_build_dist_dsa(dtype, num_key_value_heads))
    parallelize_hsdp(dist_model, mesh=ctx.mesh_for(DENSE_DOMAIN)["dp_replicate", "dp_cp_shard", "cp_replicate"])
    copy_params_local_to_dist(local, dist_model)

    dp_size = int(ctx.mesh_for(BATCH_DOMAIN)["dp"].size())
    out_dist, loss_dist = _dist_forward_and_loss(dist_model, dist_inputs)
    (loss_dist / dp_size).backward()

    torch.testing.assert_close(out_dist, out_local, atol=2e-3, rtol=1e-2)

    sync_grads_manually(dist_model)
    check_grad_distance_all_local_dist(local, dist_model)

    assert dist_inputs.pre.grad is not None
    torch.testing.assert_close(dist_inputs.pre.grad * dp_size, local_inputs.pre.grad, atol=2e-2, rtol=2e-2)
