import pytest
import torch
from d9d.kernel.stochastic import adamw_stochastic_bf16_

from d9d_test.kernel.stochastic.adamw_step.reference_impl import adamw_step_torch


@pytest.mark.local
def test_validation_logic():
    valid_shape = (128,)
    p = torch.zeros(valid_shape, dtype=torch.bfloat16, device="cuda").contiguous()
    g = torch.zeros(valid_shape, dtype=torch.bfloat16, device="cuda").contiguous()
    m = torch.zeros(valid_shape, dtype=torch.float32, device="cuda").contiguous()
    v = torch.zeros(valid_shape, dtype=torch.float32, device="cuda").contiguous()

    with pytest.raises(ValueError, match="Shape mismatch"):
        adamw_stochastic_bf16_(
            p, torch.zeros((64,), dtype=torch.bfloat16, device="cuda"),
            m, v, 1e-3, 0.9, 0.999, 1e-8, 0.0, 1
        )

    with pytest.raises(ValueError, match="Params must be BFloat16"):
        adamw_stochastic_bf16_(
            p.float(), g.float(), m, v, 1e-3, 0.9, 0.999, 1e-8, 0.0, 1
        )

    with pytest.raises(ValueError, match="States have different dtypes"):
        adamw_stochastic_bf16_(
            p, g, m.bfloat16(), v.float(), 1e-3, 0.9, 0.999, 1e-8, 0.0, 1
        )


@pytest.mark.local
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("grad_dtype", [torch.float32, torch.bfloat16])
def test_reproducibility(state_dtype, grad_dtype):
    shape = (1024,)

    p_orig = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    g_orig = torch.randn(shape, dtype=grad_dtype, device="cuda")
    m_orig = torch.zeros(shape, dtype=state_dtype, device="cuda")
    v_orig = torch.zeros(shape, dtype=state_dtype, device="cuda")

    def run_wrapper(seed):
        gen = torch.Generator().manual_seed(seed)
        p = p_orig.clone()
        g = g_orig.clone()
        m = m_orig.clone()
        v = v_orig.clone()

        adamw_stochastic_bf16_(
            p, g, m, v,
            lr=1e-3,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.01,
            step=1,
            generator=gen
        )
        return p, m, v

    p1, m1, v1 = run_wrapper(123)
    p2, m2, v2 = run_wrapper(123)
    p3, m3, v3 = run_wrapper(999)

    # Check match for same seed
    assert torch.equal(p1, p2)
    assert torch.equal(m1, m2)
    assert torch.equal(v1, v2)

    assert not torch.equal(p1, p3)

    if state_dtype == torch.bfloat16:
        assert not torch.equal(m1, m3)
        assert not torch.equal(v1, v3)


@pytest.mark.local
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("grad_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(409006,), (128,), (50,), (1024, 10)])
def test_correctness_against_reference(state_dtype, grad_dtype, shape):
    torch.manual_seed(42)

    # Inputs
    p = torch.randn(shape, dtype=torch.float32, device="cuda")
    g = torch.randn(shape, dtype=grad_dtype, device="cuda")
    m = torch.randn(shape, dtype=torch.float32, device="cuda")
    v = torch.rand(shape, dtype=torch.float32, device="cuda")

    lr, beta1, beta2, eps, wd, step = 1e-3, 0.9, 0.999, 1e-8, 0.1, 5

    p_ref, m_ref, v_ref = adamw_step_torch(
        p,
        g,
        m,
        v,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=wd,
        step=step
    )

    p_ker = p.to(torch.bfloat16)
    g_ker = g
    m_ker = m.to(state_dtype)
    v_ker = v.to(state_dtype)

    adamw_stochastic_bf16_(
        p_ker, g_ker, m_ker, v_ker,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=wd,
        step=step,
        generator=torch.Generator().manual_seed(123)
    )

    if state_dtype == torch.float32:
        assert torch.allclose(m_ker, m_ref, atol=1e-5, rtol=1e-4)
        assert torch.allclose(v_ker, v_ref, atol=1e-5, rtol=1e-4)
    else:
        assert torch.allclose(m_ker.float(), m_ref, atol=1e-2, rtol=2e-2)
        assert torch.allclose(v_ker.float(), v_ref, atol=1e-2, rtol=2e-2)

    assert torch.allclose(p_ker.float(), p_ref, atol=1e-2, rtol=2e-2)


@pytest.mark.local
@pytest.mark.parametrize("shape", [(2_000_000,)])
def test_statistical_mean_preservation(shape):
    bf16_res = 1.0 / 128.0

    fraction = 0.25
    target_update = fraction * bf16_res

    lr = target_update

    p = torch.ones(shape, dtype=torch.bfloat16, device="cuda")
    g = torch.full(shape, -1.0, dtype=torch.bfloat16, device="cuda")  # g = -1
    m = torch.zeros_like(p, dtype=torch.float32)  # m=0
    v = torch.zeros_like(p, dtype=torch.float32)  # v=0

    adamw_stochastic_bf16_(
        p, g, m, v,
        lr=lr, beta1=0.0, beta2=0.0, eps=1e-10, weight_decay=0.0,
        step=1,
        generator=torch.Generator().manual_seed(777)
    )

    p_float = p.float()

    count_changed = (p_float > 1.0).sum().item()
    freq_changed = count_changed / sum(shape)

    assert abs(freq_changed - fraction) < 0.005

    expected_val = 1.0 + target_update
    actual_mean = p_float.mean().item()

    assert abs(actual_mean - expected_val) < 1e-4
