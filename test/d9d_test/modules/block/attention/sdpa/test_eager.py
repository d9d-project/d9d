import pytest
import torch
from d9d.module.block.attention.sdpa.config import EagerSdpaBackendConfig, SdpaParameters
from d9d.module.block.attention.sdpa.impl.eager import EagerSdpa
from torch.testing import assert_close

from d9d_test.modules.block.attention.sdpa.helpers import DEVICE, build_packing, build_qkv


def _make_backend() -> EagerSdpa:
    return EagerSdpa(EagerSdpaBackendConfig(), SdpaParameters(num_sinks=None)).to(DEVICE)


@pytest.mark.local
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("segment_lengths", [[5, 3, 8], [1, 15], [4, 4, 4, 4], [16]])
def test_packing_matches_per_segment(segment_lengths, is_causal) -> None:
    """Packed block-diagonal attention must equal running each segment independently.

    This is the ground-truth semantics of sequence packing: a token attends only within its own
    segment, so a packed row and the concatenation of separately-attended segments must coincide.
    """
    num_heads, head_dim = 4, 64
    total = sum(segment_lengths)
    scale = head_dim**-0.5
    backend = _make_backend()

    q, k, v = build_qkv(1, total, num_heads, num_heads, head_dim, torch.float32)
    packing = build_packing(segment_lengths)

    packed = backend(q, k, v, attention_mask=None, packing=packing, is_causal=is_causal, scale=scale)

    # Reference: attend each segment on its own (no packing), then concatenate.
    outputs = []
    offset = 0
    for length in segment_lengths:
        segment = slice(offset, offset + length)
        outputs.append(
            backend(
                q[:, segment],
                k[:, segment],
                v[:, segment],
                attention_mask=None,
                packing=None,
                is_causal=is_causal,
                scale=scale,
            )
        )
        offset += length
    reference = torch.cat(outputs, dim=1)

    assert packed.shape == (1, total, num_heads, head_dim)
    assert_close(packed, reference, rtol=1e-5, atol=1e-5)


@pytest.mark.local
def test_single_segment_packing_matches_causal() -> None:
    """A one-segment packed row is identical to plain causal attention over the whole row."""
    num_heads, head_dim = 4, 64
    seq_len = 12
    scale = head_dim**-0.5
    backend = _make_backend()

    q, k, v = build_qkv(1, seq_len, num_heads, num_heads, head_dim, torch.float32)

    packed = backend(q, k, v, attention_mask=None, packing=build_packing([seq_len]), is_causal=True, scale=scale)
    dense = backend(q, k, v, attention_mask=None, packing=None, is_causal=True, scale=scale)

    assert_close(packed, dense, rtol=1e-5, atol=1e-5)
