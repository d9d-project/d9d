import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # for large hidden states
        triton.Config({"ROWS_PER_BLOCK": 1}, num_warps=16),
        triton.Config({"ROWS_PER_BLOCK": 1}, num_warps=32),
        triton.Config({"ROWS_PER_BLOCK": 2}, num_warps=8),
        triton.Config({"ROWS_PER_BLOCK": 2}, num_warps=16),
        triton.Config({"ROWS_PER_BLOCK": 2}, num_warps=32),
        # for mid hidden states
        triton.Config({"ROWS_PER_BLOCK": 4}, num_warps=8),
        triton.Config({"ROWS_PER_BLOCK": 4}, num_warps=16),
        triton.Config({"ROWS_PER_BLOCK": 8}, num_warps=8),
        triton.Config({"ROWS_PER_BLOCK": 8}, num_warps=16),
        # for small hidden states
        triton.Config({"ROWS_PER_BLOCK": 16}, num_warps=16),
        triton.Config({"ROWS_PER_BLOCK": 32}, num_warps=16),
    ],
    key=["M_BUCKET", "N"],
)
@triton.jit
def _rms_norm_forward_kernel(
    x_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    out_ptr: tl.tensor,
    inv_rms_ptr: tl.tensor,
    M: int,
    M_BUCKET: int,  # for auto-tuning
    N: int,
    eps: float,
    ZERO_CENTERED: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    row_start = pid * ROWS_PER_BLOCK
    rows = tl.arange(0, ROWS_PER_BLOCK)
    r = row_start + rows
    r_mask = r < M

    # we are sure that BLOCK_SIZE_N >= N
    cols = tl.arange(0, BLOCK_SIZE_N)
    col_mask = cols < N

    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    if ZERO_CENTERED:
        weight = weight + 1.0

    mask_2d = r_mask[:, None] & col_mask[None, :]
    offsets = r[:, None] * N + cols[None, :]

    # we always compute in fp32
    x = tl.load(x_ptr + offsets, mask=mask_2d, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=1) / N
    rsqrt = tl.math.rsqrt(var + eps)

    tl.store(inv_rms_ptr + r, rsqrt, mask=r_mask)

    out = (x * rsqrt[:, None] * weight[None, :]).to(x_ptr.dtype.element_ty)

    tl.store(out_ptr + offsets, out, mask=mask_2d)


def _bucketize_m(m: int) -> int:
    if m <= 2**5:
        return 2**5
    elif m >= 2**15:
        return 2**15
    else:
        return triton.next_power_of_2(m)


def rms_norm_forward(
    x: torch.Tensor, weight: torch.Tensor, eps: float, zero_centered: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the forward pass of Root Mean Square (RMS) normalization.

    Args:
        x: Input tensor to be normalized.
        weight: Learnable 1D scaling parameters.
        eps: Small scalar added to the variance for numerical stability.
        zero_centered: If True, centers the learned weight parameter around zero
            by artificially offsetting weights by 1.0 during computation.

    Returns:
        A tuple containing:
            - The final normalized output tensor.
            - The computed inverse RMS tensor (saved for the backward pass).
    """

    if not x.is_contiguous():
        x = x.contiguous()

    if not weight.is_contiguous():
        weight = weight.contiguous()

    n_size = x.shape[-1]
    m_size = x.numel() // n_size

    out = torch.empty_like(x)

    inv_rms = torch.empty((m_size,), dtype=torch.float32, device=x.device)

    block_size_n = max(triton.next_power_of_2(n_size), 16)

    m_bucket = _bucketize_m(m_size)

    def _grid(meta: dict[str, int]) -> tuple[int, ...]:
        return (triton.cdiv(m_size, meta["ROWS_PER_BLOCK"]),)

    _rms_norm_forward_kernel[_grid](
        x,
        weight,
        out,
        inv_rms,
        M=m_size,
        M_BUCKET=m_bucket,
        N=n_size,
        eps=eps,
        ZERO_CENTERED=zero_centered,
        BLOCK_SIZE_N=block_size_n,
    )

    return out, inv_rms


@triton.autotune(
    configs=[
        # for large hidden states
        triton.Config({"ROWS_PER_BLOCK": 1}, num_warps=16),
        triton.Config({"ROWS_PER_BLOCK": 1}, num_warps=32),
        triton.Config({"ROWS_PER_BLOCK": 2}, num_warps=8),
        triton.Config({"ROWS_PER_BLOCK": 2}, num_warps=16),
        triton.Config({"ROWS_PER_BLOCK": 2}, num_warps=32),
        # for mid hidden states
        triton.Config({"ROWS_PER_BLOCK": 4}, num_warps=8),
        triton.Config({"ROWS_PER_BLOCK": 4}, num_warps=16),
        triton.Config({"ROWS_PER_BLOCK": 8}, num_warps=8),
        triton.Config({"ROWS_PER_BLOCK": 8}, num_warps=16),
        # for small hidden states
        triton.Config({"ROWS_PER_BLOCK": 16}, num_warps=16),
        triton.Config({"ROWS_PER_BLOCK": 32}, num_warps=16),
    ],
    key=["M_BUCKET", "N"],
    reset_to_zero=["grad_weight_ptr"],
)
@triton.jit
def _rms_norm_backward_kernel(
    grad_out_ptr: tl.tensor,
    x_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    inv_rms_ptr: tl.tensor,
    grad_x_ptr: tl.tensor,
    grad_weight_ptr: tl.tensor,
    M: int,
    M_BUCKET: int,
    N: int,
    ZERO_CENTERED: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    # we use persistent implementation since now we can maintain acc_dweight that is local
    # and avoid constantly writing to dweight in HBM
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)

    cols = tl.arange(0, BLOCK_SIZE_N)
    col_mask = cols < N

    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    if ZERO_CENTERED:
        weight = weight + 1.0

    acc_dweight = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)

    row_start = pid * ROWS_PER_BLOCK
    stride = num_jobs * ROWS_PER_BLOCK

    for row_idx in range(row_start, M, stride):
        rows = row_idx + tl.arange(0, ROWS_PER_BLOCK)
        r_mask = rows < M

        mask_2d = r_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * N + cols[None, :]

        dout = tl.load(grad_out_ptr + offsets, mask=mask_2d, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offsets, mask=mask_2d, other=0.0).to(tl.float32)
        inv_rms = tl.load(inv_rms_ptr + rows, mask=r_mask, other=0.0)

        x_hat = x * inv_rms[:, None]

        acc_dweight += tl.sum(dout * x_hat, axis=0)

        dy = dout * weight[None, :]
        c1 = tl.sum(dy * x_hat, axis=1) / N
        dx = inv_rms[:, None] * (dy - x_hat * c1[:, None])

        tl.store(grad_x_ptr + offsets, dx.to(x_ptr.dtype.element_ty), mask=mask_2d)

    tl.atomic_add(grad_weight_ptr + cols, acc_dweight, mask=col_mask)


def rms_norm_backward(
    grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, inv_rms: torch.Tensor, zero_centered: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the backward pass of Root Mean Square (RMS) normalization.

    Args:
        grad_output: Gradient of the objective with respect to the output tensor.
        x: The original input tensor.
        weight: The learned scaling parameters for normalization.
        inv_rms: The inverse RMS values produced during the forward pass.
        zero_centered: Whether the weights were configured as zero-centered during the forward pass.

    Returns:
        A tuple containing:
            - The gradient with respect to the input tensor.
            - The gradient with respect to the weight tensor.
    """

    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    n_size = x.shape[-1]
    m_size = x.numel() // n_size

    grad_x = torch.empty_like(x)

    # must allocate fp32 and set to exactly zero for correctness on atomic adds
    grad_weight_fp32 = torch.zeros_like(weight, dtype=torch.float32)

    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    max_workers = sm_count * 2  # oversubscribe to SMs

    block_size_n = max(triton.next_power_of_2(n_size), 16)
    m_bucket = _bucketize_m(m_size)

    def _grid(meta: dict[str, int]) -> tuple[int, ...]:
        return (min(triton.cdiv(m_size, meta["ROWS_PER_BLOCK"]), max_workers),)

    _rms_norm_backward_kernel[_grid](
        grad_output,
        x,
        weight,
        inv_rms,
        grad_x,
        grad_weight_fp32,
        M=m_size,
        M_BUCKET=m_bucket,
        N=n_size,
        ZERO_CENTERED=zero_centered,
        BLOCK_SIZE_N=block_size_n,
    )

    return grad_x, grad_weight_fp32.to(weight.dtype)
