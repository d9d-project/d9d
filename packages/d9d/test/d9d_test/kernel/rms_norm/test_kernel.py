import pytest
import torch
from d9d.kernel.normalization import rms_norm
from torch.testing import assert_close

from d9d_test.kernel.rms_norm.reference_impl import rms_norm_torch


@pytest.mark.local
@pytest.mark.parametrize("shape", [(128,), (128, 128), (1024, 1024), (111, 333), (4, 16, 2048)])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 1e-3, 1e-5),
        (torch.float16, 1.6e-2, 1e-5),
        (torch.bfloat16, 1.6e-2, 1e-5),
    ],
)
@pytest.mark.parametrize("zero_centered", [False, True])
def test_rms_norm(shape, dtype, rtol, atol, zero_centered):
    torch.manual_seed(42)

    eps = 1e-6
    n_size = shape[-1]

    # Reference tensors
    x_ref = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    weight_ref = torch.randn((n_size,), dtype=dtype, device="cuda", requires_grad=True)

    # Clones for Kernel testing
    x_k = x_ref.detach().clone().requires_grad_()
    weight_k = weight_ref.detach().clone().requires_grad_()

    grad_output = torch.randn_like(x_ref)

    # Reference Forward + Backward
    out_ref = rms_norm_torch(x_ref.float(), weight_ref.float(), eps=eps, zero_centered=zero_centered).to(x_ref.dtype)
    out_ref.backward(grad_output)

    # Kernel Forward + Backward
    out_k = rms_norm(x_k, weight_k, eps=eps, zero_centered=zero_centered)
    out_k.backward(grad_output)

    # Compare Forward outputs
    assert_close(out_k, out_ref, rtol=rtol, atol=atol)

    # Compare Backward gradients
    assert_close(x_k.grad, x_ref.grad, rtol=rtol, atol=atol)
    assert_close(weight_k.grad, weight_ref.grad, rtol=rtol, atol=atol)


@pytest.mark.local
def test_rms_norm_non_contiguous() -> None:
    torch.manual_seed(42)
    dtype = torch.float32

    x_base = torch.randn((128, 256), dtype=dtype, device="cuda")
    x_ref = x_base.clone().detach().t().requires_grad_()  # Shape is (256, 128), strides (1, 256)
    x_k = x_base.clone().detach().t().requires_grad_()

    n_size = x_ref.shape[-1]

    w_base = torch.randn((n_size * 2,), dtype=dtype, device="cuda")
    weight_ref = w_base.clone().detach()[::2].requires_grad_()  # Stride of 2
    weight_k = w_base.clone().detach()[::2].requires_grad_()

    grad_output = torch.randn_like(x_ref)

    # Process Torch Base
    out_ref = rms_norm_torch(x_ref, weight_ref)
    out_ref.backward(grad_output)

    # Process Kernel
    out_k = rms_norm(x_k, weight_k)
    out_k.backward(grad_output)

    assert not x_k.is_contiguous()
    assert not weight_k.is_contiguous()

    # Verifying outputs
    assert_close(out_k, out_ref, rtol=1e-5, atol=1e-5)
    assert_close(x_k.grad, x_ref.grad, rtol=1e-5, atol=1e-5)
    assert_close(weight_k.grad, weight_ref.grad, rtol=1e-5, atol=1e-5)
