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
    restore_value=["p_ptr", "m_ptr", "v_ptr"]
)
@triton.jit
def _adamw_stochastic_bf16_kernel(
        p_ptr: tl.tensor,  # Pointer to parameters (Always BF16 -> read/write)
        g_ptr: tl.tensor,  # Pointer to gradients  (BF16 or FP32 -> read only)
        m_ptr: tl.tensor,  # Pointer to exp_avg    (BF16 or FP32 -> read/write)
        v_ptr: tl.tensor,  # Pointer to exp_avg_sq (BF16 or FP32 -> read/write)
        n_elements: int,  # Total number of elements
        lr: float,  # Learning rate
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        step: int,  # Current step (for bias correction)
        seed: int,  # Random seed for stochastic rounding
        BLOCK_SIZE: tl.constexpr,
        GRAD_IS_BF16: tl.constexpr,  # noqa: N803
        STATE_IS_BF16: tl.constexpr  # noqa: N803
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load parameters
    p_bf16 = tl.load(p_ptr + offsets, mask=mask)
    p_fp32 = p_bf16.to(tl.float32)

    # load grad
    if GRAD_IS_BF16:
        g_fp32 = tl.load(g_ptr + offsets, mask=mask).to(tl.float32)
    else:
        g_fp32 = tl.load(g_ptr + offsets, mask=mask)

    # load states
    if STATE_IS_BF16:
        m_curr = tl.load(m_ptr + offsets, mask=mask).to(tl.float32)
        v_curr = tl.load(v_ptr + offsets, mask=mask).to(tl.float32)
    else:
        m_curr = tl.load(m_ptr + offsets, mask=mask)
        v_curr = tl.load(v_ptr + offsets, mask=mask)

    # now the math goes in fp32

    # do weight decay
    p_fp32 = p_fp32 * (1.0 - lr * weight_decay)

    # update moments
    m_next = beta1 * m_curr + (1.0 - beta1) * g_fp32
    v_next = beta2 * v_curr + (1.0 - beta2) * (g_fp32 * g_fp32)

    # bias correction
    bias_correction1 = 1.0 - tl.exp(step * tl.log(beta1))
    bias_correction2 = 1.0 - tl.exp(step * tl.log(beta2))

    m_hat = m_next / bias_correction1
    v_hat = v_next / bias_correction2

    # compute update
    update = (lr * m_hat) / (tl.sqrt(v_hat) + eps)

    p_new_fp32 = p_fp32 - update

    # and now we store...
    # p -> always stochastic fp32 -> bf16
    # states -> depending on constexprs
    p_new_bf16 = fp32_to_bf16_kernel(p_new_fp32, offsets, seed)
    tl.store(p_ptr + offsets, p_new_bf16, mask=mask)

    if STATE_IS_BF16:
        m_next_bf16 = fp32_to_bf16_kernel(m_next, offsets, seed + 42)
        v_next_bf16 = fp32_to_bf16_kernel(v_next, offsets, seed + 67)

        tl.store(m_ptr + offsets, m_next_bf16, mask=mask)
        tl.store(v_ptr + offsets, v_next_bf16, mask=mask)
    else:
        tl.store(m_ptr + offsets, m_next, mask=mask)
        tl.store(v_ptr + offsets, v_next, mask=mask)


def adamw_stochastic_bf16_(  # noqa: C901
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
        generator: torch.Generator | None = None
) -> None:
    """
    Performs a single in-place AdamW optimization step.

    It is specifically designed for scenarios where parameters are stored in BFloat16.

    To mitigate precision loss during the parameter update, it utilizes stochastic rounding when casting
    FP32 calculation results back to BFloat16.

    This function supports mixed precision for gradients and optimizer states (they can be
    either FP32 or BFloat16).

    Args:
        params: The tensor of model parameters to update. Must be BFloat16 and contiguous.
        grads: The gradient tensor.
        exp_avg: The exponential moving average of gradient values (first moment).
        exp_avg_sq: The exponential moving average of squared gradient values (second moment).
        lr: The learning rate.
        beta1: Decay rate for the first moment estimate.
        beta2: Decay rate for the second moment estimate.
        eps: Term added to the denominator to improve numerical stability.
        weight_decay: Weight decay coefficient.
        step: The current optimization step count, used for bias correction.
        generator: PyTorch random number generator used to create the seed for stochastic rounding.

    Raises:
        ValueError: If main parameters are not BFloat16, if input tensor shapes do not match,
            if input tensors are not contiguous (for those that require in-place modification),
            if the optimizer states (exp_avg, exp_avg_sq) have different dtypes.
    """

    # check shape equality
    if grads.shape != params.shape:
        raise ValueError("Shape mismatch between grads and params.")

    if exp_avg.shape != params.shape:
        raise ValueError("Shape mismatch between exp_avg state and params.")

    if exp_avg_sq.shape != params.shape:
        raise ValueError("Shape mismatch between exp_avg_sq state and params.")

    # check params
    if params.dtype != torch.bfloat16:
        raise ValueError("Params must be BFloat16 for this kernel.")

    if not params.is_contiguous():
        raise ValueError("Params must be contiguous since it is an in-place kernel.")

    # check grads
    if not grads.is_contiguous():
        grads = grads.contiguous()

    # check states
    if not exp_avg.is_contiguous():
        raise ValueError("Exp_avg state must be contiguous since it is an in-place kernel.")

    if not exp_avg_sq.is_contiguous():
        raise ValueError("Exp_avg_sq state must be contiguous since it is an in-place kernel.")

    if exp_avg.dtype != exp_avg_sq.dtype:
        raise ValueError("States have different dtypes.")

    n_elements = params.numel()

    grad_is_bf16 = (grads.dtype == torch.bfloat16)
    state_is_bf16 = (exp_avg.dtype == torch.bfloat16)

    # Generate random seed
    seed = torch.randint(
        0, 2 ** 31 - 1, (1,),
        device="cpu",
        generator=generator
    ).item()

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
        STATE_IS_BF16=state_is_bf16
    )
