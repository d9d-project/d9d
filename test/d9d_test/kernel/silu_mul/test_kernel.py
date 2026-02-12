import pytest
import torch
from d9d.kernel.swiglu import silu_mul
from torch.testing import assert_close

from d9d_test.kernel.silu_mul.reference_impl import silu_mul_torch


@pytest.mark.local
@pytest.mark.parametrize("shape", [(128,), (128, 128), (1024, 1024), (111, 333)])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 1.3e-6, 1e-5),
        (torch.float16, 1.6e-2, 1e-5),
        (torch.bfloat16, 1.6e-2, 1e-5),
    ],
)
def test_silu_mul(shape, dtype, rtol, atol):
    torch.manual_seed(42)

    x_ref = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    y_ref = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)

    # Inputs for Kernel (cloned to ensure same init)
    x_k = x_ref.detach().clone().requires_grad_()
    y_k = y_ref.detach().clone().requires_grad_()

    # Forward
    out_ref = silu_mul_torch(x_ref, y_ref)
    grad_output = torch.randn_like(out_ref)
    out_ref.backward(grad_output)

    out_k = silu_mul(x_k, y_k)
    out_k.backward(grad_output)

    assert_close(out_k, out_ref, rtol=rtol, atol=atol)
    assert_close(x_k.grad, x_ref.grad, rtol=rtol, atol=atol)
    assert_close(y_k.grad, y_ref.grad, rtol=rtol, atol=atol)


@pytest.mark.local
def test_silu_mul_errors():
    x = torch.randn((100,), device="cuda")

    # Shape mismatch
    with pytest.raises(ValueError, match="same shape"):
        y_bad = torch.randn((101,), device="cuda")
        silu_mul(x, y_bad)

    with pytest.raises(ValueError, match="same device"):
        y_cpu = torch.randn((100,), device="cpu")
        silu_mul(x, y_cpu)
