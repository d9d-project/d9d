import torch
from d9d.module.block.attention import SequencePacking
from d9d.module.block.attention.sdpa.config import EagerSdpaBackendConfig, SdpaParameters
from d9d.module.block.attention.sdpa.impl.eager import EagerSdpa
from d9d.module.block.attention.sdpa.protocol import SdpaBackend
from torch.testing import assert_close

DEVICE = "cuda"


def build_qkv(batch, seq_len, num_q_heads, num_kv_heads, head_dim, dtype):
    """Builds deterministic ``(query, key, value)`` tensors in BSHD layout."""
    torch.manual_seed(42)
    q = torch.randn(batch, seq_len, num_q_heads, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch, seq_len, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(batch, seq_len, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    return q, k, v


def build_packing(segment_lengths: list[int]) -> SequencePacking:
    """Builds a ``SequencePacking`` descriptor from a list of per-segment lengths."""
    cu_seqlens = torch.tensor([0, *segment_lengths], device=DEVICE, dtype=torch.int32).cumsum(0).to(torch.int32)
    return SequencePacking(cu_seqlens=cu_seqlens, max_seqlen=max(segment_lengths))


def assert_matches_eager(
    backend: SdpaBackend,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_causal: bool,
    dtype: torch.dtype,
    rtol: float,
    atol: float,
    num_sinks: int | None = None,
    window_size: tuple[int | None, int | None] = (None, None),
    use_mask: bool = False,
    check_contiguous: bool = False,
    batch: int = 2,
    seq_len: int = 128,
    segment_lengths: list[int] | None = None,
) -> None:
    """Asserts a backend matches the eager reference in forward and backward.

    Builds matching inputs for ``backend`` (run in ``dtype``) and an `EagerSdpa`
    reference (run in float32), then compares the forward output and the
    gradients of query, key, value, and the learnable sink (when present).

    When ``segment_lengths`` is given, a single packed row of shape ``(1, total, heads, dim)`` is
    built instead (``total`` being the sum of the segment lengths) and the matching
    ``SequencePacking`` descriptor is fed to both backends, exercising the block-diagonal varlen path.

    Args:
        backend: The backend under test. Must already be on ``DEVICE`` with the
            structural parameters (sinks/window) matching the keyword arguments.
        num_q_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head dimension.
        is_causal: Whether to apply causal masking.
        dtype: Dtype the backend under test runs in.
        rtol: Relative tolerance for all comparisons.
        atol: Absolute tolerance for all comparisons.
        num_sinks: Number of learnable sinks, or ``None`` to disable.
        window_size: Sliding-window size as ``(left, right)``.
        use_mask: If True, feed a random additive attention mask to both backends.
        check_contiguous: If True, assert the backend output is contiguous.
        batch: Batch size. Forced to 1 when ``segment_lengths`` is given.
        seq_len: Sequence length. Overridden by the packed total when ``segment_lengths`` is given.
        segment_lengths: Per-segment token counts for the packed (varlen) path, or ``None`` for a
            dense row.
    """
    scale = head_dim**-0.5

    packing = None
    if segment_lengths is not None:
        batch = 1
        seq_len = sum(segment_lengths)
        packing = build_packing(segment_lengths)

    q, k, v = build_qkv(batch, seq_len, num_q_heads, num_kv_heads, head_dim, dtype)

    eager = EagerSdpa(
        EagerSdpaBackendConfig(),
        SdpaParameters(num_sinks=num_sinks, window_size=window_size),
    ).to(DEVICE)

    if num_sinks is not None:
        torch.manual_seed(99)
        sink_init = torch.randn(num_sinks, device=DEVICE, dtype=dtype)
        with torch.no_grad():
            backend.sinks.copy_(sink_init)
            eager.sinks.copy_(sink_init.float())

    mask = None
    mask_ref = None
    if use_mask:
        torch.manual_seed(7)
        mask = torch.randn(batch, num_q_heads, seq_len, seq_len, device=DEVICE, dtype=dtype)
        mask_ref = mask.float()

    # Backend under test, run in its native dtype.
    q_be = q.clone().requires_grad_(True)
    k_be = k.clone().requires_grad_(True)
    v_be = v.clone().requires_grad_(True)
    out = backend(q_be, k_be, v_be, attention_mask=mask, packing=packing, is_causal=is_causal, scale=scale)

    # Eager reference, run in float32 for precision.
    q_ref = q.float().requires_grad_(True)
    k_ref = k.float().requires_grad_(True)
    v_ref = v.float().requires_grad_(True)
    ref = eager(q_ref, k_ref, v_ref, attention_mask=mask_ref, packing=packing, is_causal=is_causal, scale=scale)

    assert out.shape == (batch, seq_len, num_q_heads, head_dim)
    if check_contiguous:
        assert out.is_contiguous()
    assert_close(out.float(), ref, rtol=rtol, atol=atol)

    grad_output = torch.randn_like(ref)
    out.backward(grad_output.to(dtype))
    ref.backward(grad_output)

    assert_close(q_be.grad.float(), q_ref.grad, rtol=rtol, atol=atol)
    assert_close(k_be.grad.float(), k_ref.grad, rtol=rtol, atol=atol)
    assert_close(v_be.grad.float(), v_ref.grad, rtol=rtol, atol=atol)

    if num_sinks is not None:
        assert backend.sinks.grad is not None
        assert_close(backend.sinks.grad.float(), eager.sinks.grad, rtol=rtol, atol=atol)
