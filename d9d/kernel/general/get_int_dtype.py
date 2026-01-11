import triton
import triton.language as tl


@triton.constexpr_function
def get_int_dtype(bitwidth: int, signed: bool) -> tl.dtype:
    return tl.core.get_int_dtype(bitwidth, signed)
