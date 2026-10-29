import pytest
import torch
from d9d.module.block.attention.sdpa.config import (
    SdpaParameters,
    TorchSdpaBackendConfig,
    TorchSdpaBackendType,
)
from d9d.module.block.attention.sdpa.impl.torch_sdpa import TorchSdpa

from d9d_test.modules.block.attention.sdpa.helpers import DEVICE, assert_matches_eager

_DTYPE = torch.float32


def _make_backend(backends=None):
    return TorchSdpa(TorchSdpaBackendConfig(backends=backends), SdpaParameters(num_sinks=None)).to(DEVICE)


@pytest.mark.local
def test_rejects_sinks() -> None:
    with pytest.raises(ValueError, match="learnable sinks"):
        TorchSdpa(TorchSdpaBackendConfig(), SdpaParameters(num_sinks=4))


@pytest.mark.local
def test_rejects_window() -> None:
    with pytest.raises(ValueError, match="sliding window"):
        TorchSdpa(TorchSdpaBackendConfig(), SdpaParameters(num_sinks=None, window_size=(8, 0)))


@pytest.mark.local
@pytest.mark.parametrize(
    ("num_q_heads", "num_kv_heads"),
    [(8, 8), (8, 2), (8, 1)],
)
# PyTorch SDPA forbids an explicit mask together with is_causal=True, so that
# combination is excluded.
@pytest.mark.parametrize(
    ("is_causal", "use_mask"),
    [(True, False), (False, False), (False, True)],
)
def test_matches_eager(num_q_heads, num_kv_heads, is_causal, use_mask) -> None:
    assert_matches_eager(
        _make_backend(),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        is_causal=is_causal,
        dtype=_DTYPE,
        rtol=1e-4,
        atol=1e-4,
        use_mask=use_mask,
        check_contiguous=True,
    )


_BACKEND_TO_ATEN_OP = {
    TorchSdpaBackendType.MATH: "aten::_scaled_dot_product_attention_math",
    TorchSdpaBackendType.FLASH_ATTENTION: "aten::_scaled_dot_product_flash_attention",
    TorchSdpaBackendType.EFFICIENT_ATTENTION: "aten::_scaled_dot_product_efficient_attention",
    TorchSdpaBackendType.CUDNN_ATTENTION: "aten::_scaled_dot_product_cudnn_attention",
}


@pytest.mark.local
@pytest.mark.parametrize("backend_type", list(TorchSdpaBackendType))
def test_backends(backend_type) -> None:
    backend = _make_backend(backends=[backend_type])

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        assert_matches_eager(
            backend.to(DEVICE),
            num_q_heads=8,
            num_kv_heads=8,
            head_dim=128,
            is_causal=True,
            dtype=torch.bfloat16,
            rtol=5e-2,
            atol=5e-2,
            check_contiguous=True,
        )

    op_names = {event.name for event in prof.events()}
    assert _BACKEND_TO_ATEN_OP[backend_type] in op_names
