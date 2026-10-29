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

    val_fp32 = tl.load(source_ptr + offsets, mask=mask)

    val_bf16 = fp32_to_bf16_kernel(val_fp32=val_fp32, offsets=offsets, seed=seed)

    tl.store(target_ptr + offsets, val_bf16, mask=mask)


def copy_fp32_to_bf16_stochastic_(
    target: torch.Tensor, source: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Copies an fp32 tensor into a bf16 tensor in place with stochastic rounding.

    Stochastic rounding rounds each value up or down at random. The probability of rounding up equals the
    fraction that the cast drops, so the expected value is preserved: ``E[round(x)] = x``. Small updates
    then accumulate in bf16 instead of being lost.

    Args:
        target: Output tensor. Must be bf16 and contiguous.
        source: Input tensor. Must be fp32 and have the same shape as ``target``.
        generator: Random number generator that seeds the rounding noise. If ``None``, the default PyTorch
            generator is used.

    Returns:
        The ``target`` tensor.

    Raises:
        ValueError: If ``target`` is not contiguous, if the shapes differ, or if ``source`` is not fp32 or
            ``target`` is not bf16.
    """
    if not source.is_contiguous():
        source = source.contiguous()

    if not target.is_contiguous():
        raise ValueError("target must be contiguous because the copy writes it in place.")

    if source.shape != target.shape:
        raise ValueError(f"source shape ({tuple(source.shape)}) must match target shape ({tuple(target.shape)}).")

    if source.dtype != torch.float32:
        raise ValueError(f"source dtype ({source.dtype}) must be torch.float32.")
    if target.dtype != torch.bfloat16:
        raise ValueError(f"target dtype ({target.dtype}) must be torch.bfloat16.")

    n_elements = source.numel()

    # A new seed for each launch, so repeated copies get different noise.
    seed = torch.randint(0, 2**31 - 1, (1,), device="cpu", generator=generator).item()

    def _grid(meta: dict[str, int]) -> tuple[int, ...]:
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _copy_fp32_to_bf16_kernel[_grid](source, target, n_elements, seed)
    return target
