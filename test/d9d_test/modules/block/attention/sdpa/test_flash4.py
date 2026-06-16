import pytest
import torch
from d9d.module.block.attention.sdpa.config import (
    FlashAttention4SdpaBackendConfig,
    SdpaParameters,
)

from d9d_test.modules.block.attention.sdpa.helpers import DEVICE, assert_matches_eager, build_qkv

pytest.importorskip("flash_attn.cute")

from d9d.module.block.attention.sdpa.impl.flash4 import FlashAttention4Sdpa


def _make_backend(num_sinks, window_size, dtype):
    backend = FlashAttention4Sdpa(
        FlashAttention4SdpaBackendConfig(),
        SdpaParameters(num_sinks=num_sinks, window_size=window_size),
    )
    return backend.to(device=DEVICE, dtype=dtype)


@pytest.mark.local
def test_sinks_parameter_creation() -> None:
    with_sinks = FlashAttention4Sdpa(FlashAttention4SdpaBackendConfig(), SdpaParameters(num_sinks=8))
    assert with_sinks.sinks is not None
    assert with_sinks.sinks.shape == (8,)

    without = FlashAttention4Sdpa(FlashAttention4SdpaBackendConfig(), SdpaParameters(num_sinks=None))
    assert without.sinks is None


@pytest.mark.local
def test_rejects_explicit_mask() -> None:
    batch, seq_len, num_heads, head_dim = 1, 8, 2, 128
    q, k, v = build_qkv(batch, seq_len, num_heads, num_heads, head_dim, torch.bfloat16)
    backend = FlashAttention4Sdpa(FlashAttention4SdpaBackendConfig(), SdpaParameters(num_sinks=None))
    mask = torch.zeros(batch, num_heads, seq_len, seq_len, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="attention mask"):
        backend(q, k, v, attention_mask=mask, is_causal=True, scale=head_dim**-0.5)


@pytest.mark.local
@pytest.mark.parametrize(
    ("num_q_heads", "num_kv_heads", "head_dim"),
    [
        (8, 8, 128),  # MHA
        (8, 2, 128),  # GQA
        (8, 1, 128),  # MQA
        (8, 8, 80),  # unaligned head_dim, exercises pad/unpad path
    ],
)
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("use_sink", [True, False])
@pytest.mark.parametrize("window_size", [(None, None), (64, 0)])
def test_matches_eager(num_q_heads, num_kv_heads, head_dim, window_size, use_sink, is_causal) -> None:
    dtype = torch.bfloat16
    num_sinks = num_q_heads if use_sink else None
    backend = _make_backend(num_sinks, window_size, dtype)

    assert_matches_eager(
        backend,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_causal=is_causal,
        dtype=dtype,
        rtol=5e-2,
        atol=5e-2,
        num_sinks=num_sinks,
        window_size=window_size,
    )
