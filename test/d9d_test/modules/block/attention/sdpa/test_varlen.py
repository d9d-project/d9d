import pytest
import torch
import torch.nn.functional as F
from d9d.module.block.attention.sdpa import (
    EagerSdpaBackendConfig,
    FlashAttention2SdpaBackendConfig,
    SdpaParameters,
    TorchSdpaBackendConfig,
    build_varlen_sdpa_backend,
)
from d9d.module.block.attention.sdpa.impl.torch_sdpa_varlen import TorchVarlenSdpa
from d9d.module.block.attention.sdpa.protocol import VarlenSdpaBackend
from torch.testing import assert_close

_DEVICE = "cuda"
_SEGMENT_LENS = (17, 64, 3, 44)
_NUM_Q_HEADS = 8
_HEAD_DIM = 64


def _build_packed_qkv(num_kv_heads: int, dtype: torch.dtype):
    torch.manual_seed(42)
    total = sum(_SEGMENT_LENS)
    q = torch.randn(total, _NUM_Q_HEADS, _HEAD_DIM, device=_DEVICE, dtype=dtype)
    k = torch.randn(total, num_kv_heads, _HEAD_DIM, device=_DEVICE, dtype=dtype)
    v = torch.randn(total, num_kv_heads, _HEAD_DIM, device=_DEVICE, dtype=dtype)
    return q, k, v


def _cu_seqlens() -> torch.Tensor:
    lens = torch.tensor((0, *_SEGMENT_LENS), device=_DEVICE, dtype=torch.int32)
    return lens.cumsum(dim=0, dtype=torch.int32)


def _per_segment_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool, scale: float
) -> torch.Tensor:
    """Computes attention independently per segment in float32 as the ground truth."""
    outputs = []
    offset = 0
    for seg_len in _SEGMENT_LENS:
        q_seg = q[offset : offset + seg_len].float().transpose(0, 1).unsqueeze(0)
        k_seg = k[offset : offset + seg_len].float().transpose(0, 1).unsqueeze(0)
        v_seg = v[offset : offset + seg_len].float().transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q_seg,
            k_seg,
            v_seg,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=q_seg.shape[1] != k_seg.shape[1],
        )
        outputs.append(out.squeeze(0).transpose(0, 1))
        offset += seg_len
    return torch.cat(outputs)


def _assert_matches_reference(
    backend: VarlenSdpaBackend,
    num_kv_heads: int,
    is_causal: bool,
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    scale = _HEAD_DIM**-0.5
    q, k, v = _build_packed_qkv(num_kv_heads, dtype)

    q_be = q.clone().requires_grad_(True)
    k_be = k.clone().requires_grad_(True)
    v_be = v.clone().requires_grad_(True)
    out = backend(
        q_be,
        k_be,
        v_be,
        cu_seqlens=_cu_seqlens(),
        max_seqlen=max(_SEGMENT_LENS),
        is_causal=is_causal,
        scale=scale,
    )

    q_ref = q.clone().requires_grad_(True)
    k_ref = k.clone().requires_grad_(True)
    v_ref = v.clone().requires_grad_(True)
    ref = _per_segment_reference(q_ref, k_ref, v_ref, is_causal=is_causal, scale=scale)

    assert out.shape == (sum(_SEGMENT_LENS), _NUM_Q_HEADS, _HEAD_DIM)
    assert_close(out.float(), ref, rtol=rtol, atol=atol)

    grad_output = torch.randn_like(ref)
    out.backward(grad_output.to(dtype))
    ref.backward(grad_output)

    assert_close(q_be.grad.float(), q_ref.grad.float(), rtol=rtol, atol=atol)
    assert_close(k_be.grad.float(), k_ref.grad.float(), rtol=rtol, atol=atol)
    assert_close(v_be.grad.float(), v_ref.grad.float(), rtol=rtol, atol=atol)


@pytest.mark.local
@pytest.mark.parametrize("num_kv_heads", [8, 2])
@pytest.mark.parametrize("is_causal", [False, True])
def test_torch_varlen_matches_per_segment(num_kv_heads: int, is_causal: bool) -> None:
    backend = build_varlen_sdpa_backend(
        params=SdpaParameters(num_sinks=None),
        backend_config=TorchSdpaBackendConfig(),
    )
    assert isinstance(backend, TorchVarlenSdpa)

    _assert_matches_reference(
        backend.to(_DEVICE),
        num_kv_heads=num_kv_heads,
        is_causal=is_causal,
        dtype=torch.float32,
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.local
@pytest.mark.parametrize("num_kv_heads", [8, 2])
def test_flash2_varlen_matches_per_segment(num_kv_heads: int) -> None:
    pytest.importorskip("flash_attn")

    backend = build_varlen_sdpa_backend(
        params=SdpaParameters(num_sinks=None),
        backend_config=FlashAttention2SdpaBackendConfig(),
    )

    _assert_matches_reference(
        backend.to(_DEVICE),
        num_kv_heads=num_kv_heads,
        is_causal=False,
        dtype=torch.bfloat16,
        rtol=1e-2,
        atol=2e-2,
    )


@pytest.mark.local
def test_varlen_rejects_eager() -> None:
    with pytest.raises(ValueError, match="eager"):
        build_varlen_sdpa_backend(
            params=SdpaParameters(num_sinks=None),
            backend_config=EagerSdpaBackendConfig(),
        )


@pytest.mark.local
def test_varlen_rejects_sinks() -> None:
    with pytest.raises(ValueError, match="sinks"):
        build_varlen_sdpa_backend(
            params=SdpaParameters(num_sinks=4),
            backend_config=TorchSdpaBackendConfig(),
        )


@pytest.mark.local
def test_varlen_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("D9D_BACKEND_AUTO_SDPA_VARLEN", '{"kind": "torch"}')

    backend = build_varlen_sdpa_backend(params=SdpaParameters(num_sinks=None), backend_config=None)

    assert isinstance(backend, TorchVarlenSdpa)


@pytest.mark.local
def test_varlen_auto_detect_falls_back_to_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    from d9d.module.block.attention.sdpa import factory as factory_mod

    monkeypatch.delenv("D9D_BACKEND_AUTO_SDPA_VARLEN", raising=False)
    monkeypatch.setattr(factory_mod.importlib.util, "find_spec", lambda name: None)

    backend = build_varlen_sdpa_backend(params=SdpaParameters(num_sinks=None), backend_config=None)

    assert isinstance(backend, TorchVarlenSdpa)
