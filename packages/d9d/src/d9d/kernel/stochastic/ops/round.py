import triton
import triton.language as tl


@triton.jit
def fp32_to_bf16_kernel(
    val_fp32: tl.tensor,
    offsets: tl.tensor,
    seed: int,
) -> tl.tensor:
    """Rounds fp32 values to bf16 stochastically.

    Args:
        val_fp32: Values to round.
        offsets: Element offsets. Each offset selects the random value for its element.
        seed: Random seed.

    Returns:
        The rounded bf16 values.
    """
    val_ui32 = val_fp32.to(tl.uint32, bitcast=True)

    # Random noise in the low 16 bits, which bf16 drops.
    rand_val = tl.randint(seed, offsets)
    noise = rand_val.to(tl.uint32) & 0xFFFF

    # Uniform noise before truncation rounds up with probability equal to the dropped fraction.
    val_ui32_noisy = val_ui32 + noise

    # The upper 16 bits of an fp32 value are its bf16 bit pattern.
    bf16_bits = (val_ui32_noisy >> 16).to(tl.int16)
    return bf16_bits.to(tl.bfloat16, bitcast=True)
