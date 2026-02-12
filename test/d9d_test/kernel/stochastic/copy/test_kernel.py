import pytest
import torch
from d9d.kernel.stochastic import copy_fp32_to_bf16_stochastic_


@pytest.mark.local
def test_bad_shape():
    src = torch.empty(100, dtype=torch.float32, device="cuda")
    tgt = torch.empty(200, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(ValueError, match="different shape"):
        copy_fp32_to_bf16_stochastic_(tgt, src)


@pytest.mark.local
def test_bad_target_contiguity():
    src = torch.empty((100, 200), dtype=torch.float32, device="cuda")
    tgt = torch.empty((200, 100), dtype=torch.bfloat16, device="cuda").T

    with pytest.raises(ValueError, match="contiguous"):
        copy_fp32_to_bf16_stochastic_(tgt, src)


@pytest.mark.local
@pytest.mark.parametrize("shape", [(10,), (100,), (128,), (1024,), (32, 32), (2048, 4096)])
def test_reproducibility_with_generator(shape):
    src = torch.randn(shape, dtype=torch.float32, device="cuda")

    # Same Seed -> Exact Match
    gen1 = torch.Generator().manual_seed(12345)
    tgt1 = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    copy_fp32_to_bf16_stochastic_(tgt1, src, generator=gen1)

    gen2 = torch.Generator().manual_seed(12345)
    tgt2 = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    copy_fp32_to_bf16_stochastic_(tgt2, src, generator=gen2)

    assert torch.equal(tgt1, tgt2)

    # Different Seed -> Different Results
    gen3 = torch.Generator().manual_seed(99999)
    tgt3 = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    copy_fp32_to_bf16_stochastic_(tgt3, src, generator=gen3)

    assert not torch.equal(tgt1, tgt3)


@pytest.mark.local
@pytest.mark.parametrize(
    "shape",
    [
        (2048, 4096),
        (2_000_000,),
    ],
)
def test_mean_preservation_bias(shape):
    """
    STATISTICAL TEST:
    If we have a value X = 1.0 + (0.3 * step), standard rounding
    will round DOWN to 1.0 (error = 0.3 steps).
    Stochastic rounding should round UP 30% of the time.
    The mean of the stochastic output should equal X.
    """
    offset_ratio = 0.30
    bf16_step = 1.0 / 128.0
    input_val = 1.0 + (offset_ratio * bf16_step)

    src = torch.full(shape, input_val, dtype=torch.float32, device="cuda")
    tgt = torch.empty(shape, dtype=torch.bfloat16, device="cuda")

    copy_fp32_to_bf16_stochastic_(tgt, src)

    # Check Stochastic Rounding
    stochastic_mean = tgt.float().mean().item()

    assert abs(stochastic_mean - input_val) < 2e-4
