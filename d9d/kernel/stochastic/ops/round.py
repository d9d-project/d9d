import triton
import triton.language as tl


@triton.jit
def fp32_to_bf16_kernel(
    val_fp32: tl.tensor,
    offsets: tl.tensor,
    seed: int,
) -> tl.tensor:
    val_ui32 = val_fp32.to(tl.uint32, bitcast=True)

    # create random noise for last bits
    rand_val = tl.randint(seed, offsets)
    noise = rand_val.to(tl.uint32) & 0xFFFF

    # add this noise (FP32)
    val_ui32_noisy = val_ui32 + noise

    # save in 16 bits
    bf16_bits = (val_ui32_noisy >> 16).to(tl.int16)
    return bf16_bits.to(tl.bfloat16, bitcast=True)
