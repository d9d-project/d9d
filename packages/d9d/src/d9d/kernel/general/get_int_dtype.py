import triton
import triton.language as tl


@triton.constexpr_function
def get_int_dtype(bitwidth: int, signed: bool) -> tl.dtype:
    """Returns the Triton integer dtype with the given bit width and signedness.

    Args:
        bitwidth: Number of bits.
        signed: Whether the dtype is signed.

    Returns:
        The Triton integer dtype.
    """
    return tl.core.get_int_dtype(bitwidth, signed)
