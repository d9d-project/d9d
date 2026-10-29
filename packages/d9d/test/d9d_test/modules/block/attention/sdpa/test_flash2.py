import pytest
import torch
from d9d.module.block.attention.sdpa.config import (
    FlashAttention2SdpaBackendConfig,
    SdpaParameters,
)

from d9d_test.modules.block.attention.sdpa.helpers import DEVICE, assert_matches_eager, build_qkv

pytest.importorskip("flash_attn")

from d9d.module.block.attention.sdpa.impl.flash2 import FlashAttention2Sdpa


@pytest.mark.local
def test_rejects_sinks() -> None:
    with pytest.raises(ValueError, match="learnable sinks"):
        FlashAttention2Sdpa(FlashAttention2SdpaBackendConfig(), SdpaParameters(num_sinks=4))


@pytest.mark.local
def test_rejects_explicit_mask() -> None:
    batch, seq_len, num_heads, head_dim = 1, 8, 2, 32
    q, k, v = build_qkv(batch, seq_len, num_heads, num_heads, head_dim, torch.bfloat16)
    backend = FlashAttention2Sdpa(FlashAttention2SdpaBackendConfig(), SdpaParameters(num_sinks=None))
    mask = torch.zeros(batch, num_heads, seq_len, seq_len, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="attention mask"):
        backend(q, k, v, attention_mask=mask, is_causal=True, scale=head_dim**-0.5)


@pytest.mark.local
@pytest.mark.parametrize(
    ("num_q_heads", "num_kv_heads"),
    [(8, 8), (8, 2), (8, 1)],
)
@pytest.mark.parametrize("window_size", [(None, None), (64, 0)])
@pytest.mark.parametrize("is_causal", [True, False])
def test_matches_eager(num_q_heads, num_kv_heads, window_size, is_causal) -> None:
    dtype = torch.bfloat16
    backend = FlashAttention2Sdpa(
        FlashAttention2SdpaBackendConfig(),
        SdpaParameters(num_sinks=None, window_size=window_size),
    )

    assert_matches_eager(
        backend,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        is_causal=is_causal,
        dtype=dtype,
        rtol=5e-2,
        atol=5e-2,
        window_size=window_size,
    )
