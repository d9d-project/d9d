import pytest
import torch
from d9d.module.block.normalization import RMSNorm
from torch import nn
from torch.testing import assert_close


@pytest.mark.local
def test_rms_norm_init() -> None:
    hidden_size = 128

    layer_standard = RMSNorm(hidden_size, zero_centered=False)
    layer_standard.reset_parameters()
    assert torch.all(layer_standard.weight == 1.0)

    layer_zero = RMSNorm(hidden_size, zero_centered=True)
    layer_zero.reset_parameters()
    assert torch.all(layer_zero.weight == 0.0)


@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.local
def test_rms_norm_forward_backward(zero_centered: bool) -> None:
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32
    eps = 1e-6

    input_data = torch.randn((4, 32, 1024), dtype=dtype, device=device)

    layer = RMSNorm(input_data.shape[-1], eps=eps, zero_centered=zero_centered).to(device=device, dtype=dtype)
    layer.reset_parameters()
    with torch.no_grad():
        layer.weight.add_(torch.randn_like(layer.weight) * 0.1)

    x = input_data.clone().requires_grad_()
    out = layer(x)

    layer_ref = nn.RMSNorm(input_data.shape[-1], eps=eps).to(device=device, dtype=dtype)
    with torch.no_grad():
        layer_ref.weight.copy_(layer.weight)
        if zero_centered:
            layer_ref.weight.add_(1.0)
    x_ref = input_data.clone().requires_grad_()
    out_ref = layer_ref(x_ref)

    assert_close(out, out_ref, rtol=1e-5, atol=1e-5)

    grad_output = torch.randn_like(out)

    out.backward(grad_output)
    out_ref.backward(grad_output)

    assert_close(x.grad, x_ref.grad, rtol=1e-5, atol=1e-5)
    assert_close(layer.weight.grad, layer_ref.weight.grad, rtol=1e-5, atol=1e-5)
