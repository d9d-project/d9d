import torch
import triton
import triton.language as tl


def _size_bucket(n_elements: int) -> int:
    # Autotune small and large inputs separately.
    if n_elements < 8192:
        return 0
    else:
        return 1


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
    ],
    key=["size_bucket"],
)
@triton.jit
def _silu_mul_kernel(
    x_ptr: torch.Tensor,
    y_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    n_elements: int,
    size_bucket: int,  # Used for autotuning
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)  # tl.sigmoid needs fp32 input
    y = tl.load(y_ptr + offsets, mask=mask)

    # Cast silu(x) to the input dtype before the multiply to match PyTorch eager results.
    silu_x = (x_fp32 * tl.sigmoid(x_fp32)).cast(y.dtype)
    out = silu_x * y

    tl.store(out_ptr + offsets, out, mask=mask)


def silu_mul_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the forward pass of ``silu(x) * y``.

    Args:
        x: Input passed through SiLU.
        y: Input multiplied by ``silu(x)``.

    Returns:
        The result, with the same shape as the inputs.

    Raises:
        ValueError: If ``x`` and ``y`` differ in shape or device.
    """
    if x.shape != y.shape or x.device != y.device:
        raise ValueError(
            f"x shape ({tuple(x.shape)}) and device ({x.device}) must match "
            f"y shape ({tuple(y.shape)}) and device ({y.device})."
        )

    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    n_elements = x.numel()
    out = torch.empty_like(x)

    def _grid(meta: dict[str, int]) -> tuple[int, ...]:
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _silu_mul_kernel[_grid](x, y, out, n_elements, size_bucket=_size_bucket(n_elements))

    return out


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
    ],
    key=["size_bucket"],
)
@triton.jit
def _silu_mul_backward_kernel(
    grad_out_ptr: torch.Tensor,
    x_ptr: torch.Tensor,
    y_ptr: torch.Tensor,
    grad_x_ptr: torch.Tensor,
    grad_y_ptr: torch.Tensor,
    n_elements: int,
    size_bucket: int,  # Used for autotuning
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    dout = tl.load(grad_out_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)  # tl.sigmoid needs fp32 input
    y = tl.load(y_ptr + offsets, mask=mask)

    # Recompute silu(x) instead of saving it in the forward pass.
    sig_x = tl.sigmoid(x)
    silu_x = x * sig_x

    # grad_y = dout * silu(x)
    dx_silu_x = dout * silu_x
    tl.store(grad_y_ptr + offsets, dx_silu_x, mask=mask)

    # silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    #          = sigmoid(x) + silu(x) * (1 - sigmoid(x))
    d_silu = sig_x + silu_x * (1.0 - sig_x)

    # grad_x = dout * y * silu'(x)
    dx = dout * y * d_silu
    tl.store(grad_x_ptr + offsets, dx, mask=mask)


def silu_mul_backward(grad_output: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the backward pass of ``silu(x) * y``.

    Args:
        grad_output: Gradient of the loss with respect to the output.
        x: Input ``x`` of the forward pass.
        y: Input ``y`` of the forward pass.

    Returns:
        A tuple of the gradients with respect to ``x`` and ``y``.
    """
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    n_elements = x.numel()

    grad_x = torch.empty_like(x)
    grad_y = torch.empty_like(y)

    def _grid(meta: dict[str, int]) -> tuple[int, ...]:
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _silu_mul_backward_kernel[_grid](
        grad_output, x, y, grad_x, grad_y, n_elements, size_bucket=_size_bucket(n_elements)
    )

    return grad_x, grad_y
