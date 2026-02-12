import torch
import triton
import triton.language as tl


def _size_bucket(n_elements: int) -> int:
    # different auto-tuning for small and asymptotically large kernels
    # perhaps we could extend this in future?
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
    size_bucket: int,  # used for autotuning
    BLOCK_SIZE: tl.constexpr,
):
    # prepare
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # read
    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)  # sigmoid wants fp32
    y = tl.load(y_ptr + offsets, mask=mask)

    # compute
    # cast back to match with torch
    silu_x = (x_fp32 * tl.sigmoid(x_fp32)).cast(y.dtype)
    out = silu_x * y

    # write
    tl.store(out_ptr + offsets, out, mask=mask)


def silu_mul_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the forward pass of silu(x)*y using Triton.

    Args:
        x: Input tensor x.
        y: Input tensor y.

    Returns:
        The output tensor.

    Raises:
        ValueError: If inputs x and y do not match in shape or device.
    """

    if x.shape != y.shape or x.device != y.device:
        raise ValueError("Inputs x and y must have the same shape, be on same device.")

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
    size_bucket: int,  # used for autotuning
    BLOCK_SIZE: tl.constexpr,
):
    # prepare
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # read
    dout = tl.load(grad_out_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)  # sigmoid wants fp32
    y = tl.load(y_ptr + offsets, mask=mask)

    # Recompute Silu components
    sig_x = tl.sigmoid(x)
    silu_x = x * sig_x

    # Compute grad_y
    # dy = dout * silu(x)
    dx_silu_x = dout * silu_x  # Reuse this variable name logic
    tl.store(grad_y_ptr + offsets, dx_silu_x, mask=mask)

    # Compute grad_x
    # silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    #          = sigmoid(x) + silu(x) * (1 - sigmoid(x))
    d_silu = sig_x + silu_x * (1.0 - sig_x)

    # dx = dout * y * silu'(x)
    dx = dout * y * d_silu
    tl.store(grad_x_ptr + offsets, dx, mask=mask)


def silu_mul_backward(grad_output: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the backward pass of silu(x)*y using Triton.

    Args:
        grad_output: Gradient of the loss with respect to the output.
        x: Original input tensor x.
        y: Original input tensor y.

    Returns:
        A tuple of (grad_x, grad_y).
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
