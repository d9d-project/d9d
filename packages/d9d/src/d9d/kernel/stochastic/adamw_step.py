import torch
import triton
import triton.language as tl

from .ops import fp32_to_bf16_kernel


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
    ],
    key=["n_elements"],
    restore_value=["p_ptr", "m_ptr", "v_ptr"],
)
@triton.jit
def _adamw_stochastic_bf16_kernel(
    p_ptr: tl.tensor,  # Parameters: bf16, read/write
    g_ptr: tl.tensor,  # Gradients: bf16 or fp32, read only
    m_ptr: tl.tensor,  # exp_avg: bf16 or fp32, read/write
    v_ptr: tl.tensor,  # exp_avg_sq: bf16 or fp32, read/write
    n_elements: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    seed: int,
    BLOCK_SIZE: tl.constexpr,
    GRAD_IS_BF16: tl.constexpr,
    STATE_IS_BF16: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p_bf16 = tl.load(p_ptr + offsets, mask=mask)
    p_fp32 = p_bf16.to(tl.float32)

    if GRAD_IS_BF16:
        g_fp32 = tl.load(g_ptr + offsets, mask=mask).to(tl.float32)
    else:
        g_fp32 = tl.load(g_ptr + offsets, mask=mask)

    if STATE_IS_BF16:
        m_curr = tl.load(m_ptr + offsets, mask=mask).to(tl.float32)
        v_curr = tl.load(v_ptr + offsets, mask=mask).to(tl.float32)
    else:
        m_curr = tl.load(m_ptr + offsets, mask=mask)
        v_curr = tl.load(v_ptr + offsets, mask=mask)

    # All math below is in fp32.

    # Decoupled weight decay.
    p_fp32 = p_fp32 * (1.0 - lr * weight_decay)

    # Update moments.
    m_next = beta1 * m_curr + (1.0 - beta1) * g_fp32
    v_next = beta2 * v_curr + (1.0 - beta2) * (g_fp32 * g_fp32)

    # Bias correction.
    bias_correction1 = 1.0 - tl.exp(step * tl.log(beta1))
    bias_correction2 = 1.0 - tl.exp(step * tl.log(beta2))

    m_hat = m_next / bias_correction1
    v_hat = v_next / bias_correction2

    update = (lr * m_hat) / (tl.sqrt(v_hat) + eps)

    p_new_fp32 = p_fp32 - update

    # Parameters are always rounded to bf16 stochastically; states only if they are stored in bf16.
    p_new_bf16 = fp32_to_bf16_kernel(p_new_fp32, offsets, seed)
    tl.store(p_ptr + offsets, p_new_bf16, mask=mask)

    if STATE_IS_BF16:
        # Offset the seed so each tensor gets its own noise.
        m_next_bf16 = fp32_to_bf16_kernel(m_next, offsets, seed + 42)
        v_next_bf16 = fp32_to_bf16_kernel(v_next, offsets, seed + 67)

        tl.store(m_ptr + offsets, m_next_bf16, mask=mask)
        tl.store(v_ptr + offsets, v_next_bf16, mask=mask)
    else:
        tl.store(m_ptr + offsets, m_next, mask=mask)
        tl.store(v_ptr + offsets, v_next, mask=mask)


def adamw_stochastic_bf16_(  # noqa: C901 - a flat list of input checks before one kernel launch
    params: torch.Tensor,
    grads: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    generator: torch.Generator | None = None,
) -> None:
    """Performs one in-place AdamW step on bf16 parameters with stochastic rounding.

    The update is computed in fp32, and the new parameters are rounded to bf16 stochastically. Gradients and
    optimizer states can be fp32 or bf16. States stored in bf16 are also rounded stochastically.

    Args:
        params: Parameters to update in place. Must be bf16 and contiguous.
        grads: Gradients, in bf16 or fp32.
        exp_avg: First moment (moving average of the gradients), updated in place. Must be contiguous.
        exp_avg_sq: Second moment (moving average of the squared gradients), updated in place. Must be
            contiguous and have the same dtype as ``exp_avg``.
        lr: Learning rate.
        beta1: Decay rate of the first moment.
        beta2: Decay rate of the second moment.
        eps: Term added to the denominator for numerical stability.
        weight_decay: Decoupled weight decay coefficient.
        step: Number of the current step, used for bias correction. Must be at least 1.
        generator: Random number generator that seeds the stochastic rounding. If ``None``, the default
            PyTorch generator is used.

    Raises:
        ValueError: If ``grads``, ``exp_avg`` or ``exp_avg_sq`` differs in shape from ``params``, if
            ``params`` is not bf16, if ``params``, ``exp_avg`` or ``exp_avg_sq`` is not contiguous, or if
            ``exp_avg`` and ``exp_avg_sq`` have different dtypes.
    """
    if grads.shape != params.shape:
        raise ValueError(f"grads shape ({tuple(grads.shape)}) must match params shape ({tuple(params.shape)}).")

    if exp_avg.shape != params.shape:
        raise ValueError(f"exp_avg shape ({tuple(exp_avg.shape)}) must match params shape ({tuple(params.shape)}).")

    if exp_avg_sq.shape != params.shape:
        raise ValueError(
            f"exp_avg_sq shape ({tuple(exp_avg_sq.shape)}) must match params shape ({tuple(params.shape)})."
        )

    if params.dtype != torch.bfloat16:
        raise ValueError(f"params dtype ({params.dtype}) must be torch.bfloat16.")

    if not params.is_contiguous():
        raise ValueError("params must be contiguous because the kernel updates it in place.")

    if not grads.is_contiguous():
        grads = grads.contiguous()

    if not exp_avg.is_contiguous():
        raise ValueError("exp_avg must be contiguous because the kernel updates it in place.")

    if not exp_avg_sq.is_contiguous():
        raise ValueError("exp_avg_sq must be contiguous because the kernel updates it in place.")

    if exp_avg.dtype != exp_avg_sq.dtype:
        raise ValueError(f"exp_avg dtype ({exp_avg.dtype}) and exp_avg_sq dtype ({exp_avg_sq.dtype}) must match.")

    n_elements = params.numel()

    grad_is_bf16 = grads.dtype == torch.bfloat16
    state_is_bf16 = exp_avg.dtype == torch.bfloat16

    seed = torch.randint(0, 2**31 - 1, (1,), device="cpu", generator=generator).item()

    def _grid(meta: dict[str, int]) -> tuple[int, ...]:
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _adamw_stochastic_bf16_kernel[_grid](
        params,
        grads,
        exp_avg,
        exp_avg_sq,
        n_elements,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        step,
        seed,
        GRAD_IS_BF16=grad_is_bf16,
        STATE_IS_BF16=state_is_bf16,
    )
