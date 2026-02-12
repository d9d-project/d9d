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
)
@triton.jit
def _copy_fp32_to_bf16_kernel(
    source_ptr: torch.Tensor, target_ptr: torch.Tensor, n_elements: int, seed: int, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load source value (fp32)
    val_fp32 = tl.load(source_ptr + offsets, mask=mask)

    val_bf16 = fp32_to_bf16_kernel(val_fp32=val_fp32, offsets=offsets, seed=seed)

    tl.store(target_ptr + offsets, val_bf16, mask=mask)


def copy_fp32_to_bf16_stochastic_(
    target: torch.Tensor, source: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """
    Copies elements from a Float32 tensor to a BFloat16 tensor using stochastic rounding.

    Unlike standard round-to-nearest casting, stochastic rounding probabilistically rounds
    numbers up or down based on the value of the bits being truncated. This preserves the
    expected value of the tensor (E[round(x)] = x), which is crucial for accumulating
    gradients or parameters in low precision without stagnation.

    This operation is performed in-place on the target tensor.

    Args:
        target: The output tensor where results are written. Must be of type BFloat16
            and contiguous.
        source: The input tensor containing values to copy. Must be of type Float32.
        generator: An optional PyTorch RNG generator to strictly control the random
            noise used for rounding.

    Returns:
        The target tensor, modified in-place.

    Raises:
        ValueError: If target is not contiguous, if source/target shapes do not match,
            or if dtypes are not FP32 and BF16 respectively.
    """

    if not source.is_contiguous():
        source = source.contiguous()

    if not target.is_contiguous():
        raise ValueError("Since this is an in-place operation, target should be a contiguous tensor!")

    if source.shape != target.shape:
        raise ValueError("Source and Target Tensors are of different shapes")

    if source.dtype != torch.float32:
        raise ValueError("Source must be Float32")
    if target.dtype != torch.bfloat16:
        raise ValueError("Target must be BFloat16")

    n_elements = source.numel()

    # Generate a random seed for this specific kernel launch
    seed = torch.randint(0, 2**31 - 1, (1,), device="cpu", generator=generator).item()

    def _grid(meta: dict[str, int]) -> tuple[int, ...]:
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _copy_fp32_to_bf16_kernel[_grid](source, target, n_elements, seed)
    return target
