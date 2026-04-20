"""Tests for d9d's flash_attn wrapper (fixed-length and varlen)."""

import pytest
import torch
from d9d.kernel.flash_attn import flash_attn_func, flash_attn_varlen_func
from torch.testing import assert_close

from d9d_test.kernel.flash_attn.reference_impl import eager_sink_attention


def _build_inputs(batch, seq_len, num_q_heads, num_kv_heads, head_dim, dtype):
    torch.manual_seed(42)
    q = torch.randn(batch, seq_len, num_q_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seq_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seq_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    return q, k, v


def _init_sinks(num_heads, dtype):
    torch.manual_seed(99)
    return torch.randn(num_heads, device="cuda", dtype=dtype)


@pytest.mark.local
@pytest.mark.parametrize(
    ("num_q_heads", "num_kv_heads", "window_size"),
    [
        (8, 8, (None, None)),  # MHA, full causal
        (8, 2, (None, None)),  # GQA, full causal
        (8, 1, (None, None)),  # MQA, full causal
        (8, 8, (64, 0)),  # MHA, sliding window
        (8, 2, (64, 0)),  # GQA, sliding window
        (8, 1, (64, 0)),  # MQA, sliding window
    ],
)
@pytest.mark.parametrize("use_sink", [True, False])
def test_attention(num_q_heads, num_kv_heads, window_size, use_sink):
    batch, seq_len, head_dim = 2, 128, 128
    dtype = torch.bfloat16
    scale = head_dim**-0.5

    q, k, v = _build_inputs(batch, seq_len, num_q_heads, num_kv_heads, head_dim, dtype)
    sink_init = _init_sinks(num_q_heads, dtype) if use_sink else None

    # --- Eager reference (float32 for precision) ---
    q_ref = q.float().requires_grad_(True)
    k_ref = k.float().requires_grad_(True)
    v_ref = v.float().requires_grad_(True)
    sink_ref = sink_init.float().requires_grad_(True) if use_sink else None

    out_ref = eager_sink_attention(q_ref, k_ref, v_ref, sink_ref, scale, causal=True, window_size=window_size)
    out_ref.sum().backward()

    # --- FA4 wrapper ---
    q_fa = q.clone().requires_grad_(True)
    k_fa = k.clone().requires_grad_(True)
    v_fa = v.clone().requires_grad_(True)
    sink_fa = sink_init.clone().requires_grad_(True) if use_sink else None

    out_fa, _ = flash_attn_func(
        q_fa, k_fa, v_fa, softmax_scale=scale, causal=True, window_size=window_size, learnable_sink=sink_fa
    )
    out_fa.sum().backward()

    # --- Compare forward ---
    assert_close(out_fa.float(), out_ref, atol=5e-2, rtol=5e-2)

    # --- Compare Q, K, V grads ---
    assert_close(q_fa.grad.float(), q_ref.grad, atol=5e-2, rtol=5e-2)
    assert_close(k_fa.grad.float(), k_ref.grad, atol=5e-2, rtol=5e-2)
    assert_close(v_fa.grad.float(), v_ref.grad, atol=5e-2, rtol=5e-2)

    # --- Compare sink gradient ---
    if use_sink:
        assert sink_fa.grad is not None, "sink.grad must not be None"
        assert_close(sink_fa.grad.float(), sink_ref.grad, atol=5e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# Varlen tests
# ---------------------------------------------------------------------------


def _build_varlen_inputs(seqlens, num_q_heads, num_kv_heads, head_dim, dtype):
    """Build packed varlen tensors from a list of sequence lengths."""
    torch.manual_seed(42)
    total_q = sum(seqlens)
    q = torch.randn(total_q, num_q_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(total_q, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(total_q, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    cu_seqlens = torch.zeros(len(seqlens) + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = torch.tensor(seqlens, device="cuda", dtype=torch.int32).cumsum(0)
    max_seqlen = max(seqlens)
    return q, k, v, cu_seqlens, max_seqlen


@pytest.mark.local
def test_varlen_no_sink():
    """Varlen forward+backward works without sinks."""
    seqlens = [64, 128, 32]
    num_heads, head_dim = 8, 128
    dtype = torch.bfloat16
    scale = head_dim**-0.5

    q, k, v, cu, max_s = _build_varlen_inputs(seqlens, num_heads, num_heads, head_dim, dtype)
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)

    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max_s,
        max_seqlen_k=max_s,
        softmax_scale=scale,
        causal=True,
    )
    out.sum().backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


@pytest.mark.local
def test_varlen_sink_gradient_nonzero():
    """Varlen sink gradient is non-trivial."""
    seqlens = [64, 128, 32]
    num_heads, head_dim = 8, 128
    dtype = torch.bfloat16
    scale = head_dim**-0.5

    q, k, v, cu, max_s = _build_varlen_inputs(seqlens, num_heads, num_heads, head_dim, dtype)
    sink = _init_sinks(num_heads, dtype).requires_grad_(True)

    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max_s,
        max_seqlen_k=max_s,
        softmax_scale=scale,
        causal=True,
        learnable_sink=sink,
    )
    out.sum().backward()

    assert sink.grad is not None
    assert sink.grad.abs().sum() > 0, "varlen sink gradient should be non-zero"
