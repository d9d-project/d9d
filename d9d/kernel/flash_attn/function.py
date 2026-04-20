from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd  # noqa: PLC2701

if TYPE_CHECKING:

    class _Ctx(torch.autograd.function.FunctionCtx):
        saved_tensors: tuple[torch.Tensor, ...]
        softmax_scale: float | None
        causal: bool
        window_size: tuple[int | None, int | None]
        softcap: float
        deterministic: bool
        return_lse: bool
        has_learnable_sink: bool

    class _VarlenCtx(_Ctx):
        cu_seqlens_q: torch.Tensor | None
        cu_seqlens_k: torch.Tensor | None
        seqused_q: torch.Tensor | None
        seqused_k: torch.Tensor | None
        max_seqlen_q: int | None
        max_seqlen_k: int | None


# ---------------------------------------------------------------------------
# Shared dsink computation
# ---------------------------------------------------------------------------


def _compute_dsink(
    dout: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    learnable_sink: torch.Tensor,
) -> torch.Tensor:
    """Analytic gradient for ``learnable_sink``.

    The sink forward adds ``exp(sink)`` to the softmax denominator::

        out_i = Σ_j [exp(a_ij) / (L_i + exp(s))] · v_j

    where ``lse = log(L_i + exp(s))``.  Differentiating::

        dsink_h = -Σ_{b,i} exp(s_h - lse_{b,h,i}) · (dout · out)_{b,i,h}

    Handles both batched ``(B, S, H, D)`` and varlen ``(total, H, D)`` layouts.
    """
    # dot product of dout and out per position, summed over head_dim
    dot_do = (dout.float() * out.float()).sum(dim=-1)  # (B, S, H) or (total, H)

    if lse.dim() == 3:
        # Batched: lse is (B, H, S), dot_do is (B, S, H)
        dot_do = dot_do.transpose(1, 2)  # (B, H, S)
        neg_gate = torch.exp(learnable_sink.float()[None, :, None] - lse.float())
        return -(neg_gate * dot_do).sum(dim=(0, 2)).to(learnable_sink.dtype)
    else:
        # Varlen: lse is (H, total), dot_do is (total, H)
        dot_do = dot_do.transpose(0, 1)  # (H, total)
        neg_gate = torch.exp(learnable_sink.float()[:, None] - lse.float())
        return -(neg_gate * dot_do).sum(dim=1).to(learnable_sink.dtype)


# ---------------------------------------------------------------------------
# Fixed-length attention
# ---------------------------------------------------------------------------


class _FlashAttnFunc(torch.autograd.Function):
    """Autograd wrapper around FA4's low-level forward/backward.

    Adds the ``learnable_sink`` gradient that FA4 does not yet compute.
    When no sink is used, this is a thin passthrough to the FA4 kernels.
    """

    @staticmethod
    def forward(
        ctx: _Ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None,
        causal: bool,
        window_size: tuple[int | None, int | None],
        learnable_sink: torch.Tensor | None,
        softcap: float,
        num_splits: int,
        pack_gqa: bool | None,
        deterministic: bool,
        return_lse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run FA4 forward, optionally pre-allocating LSE for sink gradients.

        When ``learnable_sink`` is provided, LSE is always materialised
        (even if ``q``/``k``/``v`` don't require grad) because the backward
        needs it for the analytic ``dsink`` computation.  Otherwise the
        allocation is left to FA4's internal logic.
        """
        lse_buf = None
        if learnable_sink is not None:
            batch_size, seqlen, num_heads = q.shape[0], q.shape[1], q.shape[-2]
            lse_buf = torch.empty(batch_size, num_heads, seqlen, dtype=torch.float32, device=q.device)

        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            return_lse=return_lse,
            lse=lse_buf,
        )
        if learnable_sink is not None:
            ctx.save_for_backward(q, k, v, out, lse, learnable_sink)
        else:
            ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.return_lse = return_lse
        ctx.has_learnable_sink = learnable_sink is not None
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: _Ctx,
        dout: torch.Tensor,
        dlse: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, torch.Tensor | None, None, None, None, None, None
    ]:
        """Run FA4 backward for ``dq``/``dk``/``dv``, then compute ``dsink`` analytically."""
        if ctx.has_learnable_sink:
            q, k, v, out, lse, learnable_sink = ctx.saved_tensors
        else:
            q, k, v, out, lse = ctx.saved_tensors
            learnable_sink = None

        if not ctx.return_lse:
            dlse = None
        if dout is None:
            dout = torch.zeros_like(out)

        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            window_size_left=ctx.window_size[0],
            window_size_right=ctx.window_size[1],
            deterministic=ctx.deterministic,
            dlse=dlse,
        )

        dsink = _compute_dsink(dout, out, lse, learnable_sink) if learnable_sink is not None else None
        return dq, dk, dv, None, None, None, dsink, None, None, None, None, None


# ---------------------------------------------------------------------------
# Variable-length attention
# ---------------------------------------------------------------------------


class _FlashAttnVarlenFunc(torch.autograd.Function):
    """Variable-length variant of :class:`_FlashAttnFunc`.

    Accepts ``cu_seqlens_q``/``cu_seqlens_k`` for packed sequences.
    Adds the ``learnable_sink`` gradient that FA4 does not yet compute.
    """

    @staticmethod
    def forward(
        ctx: _VarlenCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None,
        cu_seqlens_k: torch.Tensor | None,
        seqused_q: torch.Tensor | None,
        seqused_k: torch.Tensor | None,
        max_seqlen_q: int | None,
        max_seqlen_k: int | None,
        page_table: torch.Tensor | None,
        softmax_scale: float | None,
        causal: bool,
        window_size: tuple[int | None, int | None],
        learnable_sink: torch.Tensor | None,
        softcap: float,
        num_splits: int,
        pack_gqa: bool | None,
        deterministic: bool,
        return_lse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run FA4 varlen forward, pre-allocating LSE when sinks are active."""
        lse_buf = None
        if learnable_sink is not None:
            num_heads = q.shape[-2]
            total_q = q.shape[0]
            lse_buf = torch.empty(num_heads, total_q, dtype=torch.float32, device=q.device)

        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            return_lse=return_lse,
            lse=lse_buf,
        )
        if learnable_sink is not None:
            ctx.save_for_backward(q, k, v, out, lse, learnable_sink)
        else:
            ctx.save_for_backward(q, k, v, out, lse)
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_k = cu_seqlens_k
        ctx.seqused_q = seqused_q
        ctx.seqused_k = seqused_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.return_lse = return_lse
        ctx.has_learnable_sink = learnable_sink is not None
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: _VarlenCtx,
        dout: torch.Tensor,
        dlse: torch.Tensor | None,
    ) -> tuple:
        """Run FA4 varlen backward for ``dq``/``dk``/``dv``, then compute ``dsink``."""
        if ctx.has_learnable_sink:
            q, k, v, out, lse, learnable_sink = ctx.saved_tensors
        else:
            q, k, v, out, lse = ctx.saved_tensors
            learnable_sink = None

        if not ctx.return_lse:
            dlse = None
        if dout is None:
            dout = torch.zeros_like(out)

        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            window_size_left=ctx.window_size[0],
            window_size_right=ctx.window_size[1],
            cu_seqlens_q=ctx.cu_seqlens_q,
            cu_seqlens_k=ctx.cu_seqlens_k,
            seqused_q=ctx.seqused_q,
            seqused_k=ctx.seqused_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            deterministic=ctx.deterministic,
            dlse=dlse,
        )

        dsink = _compute_dsink(dout, out, lse, learnable_sink) if learnable_sink is not None else None

        # Return order: q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
        #   max_seqlen_q, max_seqlen_k, page_table, softmax_scale, causal, window_size,
        #   learnable_sink, softcap, num_splits, pack_gqa, deterministic, return_lse
        return dq, dk, dv, *((None,) * 10), dsink, *((None,) * 5)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int | None, int | None] = (None, None),
    learnable_sink: torch.Tensor | None = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: bool | None = None,
    deterministic: bool = False,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for ``flash_attn.cute.flash_attn_func``.

    Wraps FA4's forward and backward kernels directly.  When ``learnable_sink``
    is provided, the backward additionally computes its gradient (which FA4
    does not yet implement).  Without sinks, this is a thin passthrough.

    Args:
        q: Query tensor. Shape: ``(batch, seqlen, num_heads, head_dim)``.
        k: Key tensor. Shape: ``(batch, seqlen, num_kv_heads, head_dim)``.
        v: Value tensor. Shape: ``(batch, seqlen, num_kv_heads, head_dim)``.
        softmax_scale: Scaling factor for QK^T (default ``1/sqrt(head_dim)``).
        causal: Apply causal mask.
        window_size: ``(left, right)`` sliding window.  ``(None, None)`` for full attention.
        learnable_sink: Per-head sink logit. Shape: ``(num_heads,)``.  ``None`` to disable.
        softcap: Softcap value for clamping attention logits.
        num_splits: Number of splits for split-KV.
        pack_gqa: Pack GQA heads for efficiency.
        deterministic: Use deterministic backward.
        return_lse: Return log-sum-exp alongside the output.

    Returns:
        ``(output, lse)`` where ``lse`` may be ``None`` when not requested.
    """
    return _FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        return_lse,
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    page_table: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int | None, int | None] = (None, None),
    learnable_sink: torch.Tensor | None = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: bool | None = None,
    deterministic: bool = False,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for ``flash_attn.cute.flash_attn_varlen_func``.

    Variable-length variant that accepts ``cu_seqlens_q``/``cu_seqlens_k``
    for packed sequences.  Adds sink gradient support like :func:`flash_attn_func`.

    Args:
        q: Query tensor. Shape: ``(total_q, num_heads, head_dim)``.
        k: Key tensor. Shape: ``(total_k, num_kv_heads, head_dim)``.
        v: Value tensor. Shape: ``(total_k, num_kv_heads, head_dim)``.
        cu_seqlens_q: Cumulative sequence lengths for queries. Shape: ``(batch + 1,)``.
        cu_seqlens_k: Cumulative sequence lengths for keys. Shape: ``(batch + 1,)``.
        max_seqlen_q: Maximum query sequence length in the batch.
        max_seqlen_k: Maximum key sequence length in the batch.
        seqused_q: Actual used query lengths per sequence. Shape: ``(batch,)``.
        seqused_k: Actual used key lengths per sequence. Shape: ``(batch,)``.
        page_table: Page table for paged KV cache.
        softmax_scale: Scaling factor for QK^T (default ``1/sqrt(head_dim)``).
        causal: Apply causal mask.
        window_size: ``(left, right)`` sliding window.  ``(None, None)`` for full attention.
        learnable_sink: Per-head sink logit. Shape: ``(num_heads,)``.  ``None`` to disable.
        softcap: Softcap value for clamping attention logits.
        num_splits: Number of splits for split-KV.
        pack_gqa: Pack GQA heads for efficiency.
        deterministic: Use deterministic backward.
        return_lse: Return log-sum-exp alongside the output.

    Returns:
        ``(output, lse)`` where ``lse`` may be ``None`` when not requested.
    """
    return _FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        return_lse,
    )
