import pytest
import torch
from d9d.core.autograd import GLOBAL_GRAD_CONTEXT, GradDirection
from d9d.module.block.moe.grouped_mm import (
    AnyGroupedMmBackendConfig,
    EagerGroupedMmBackendConfig,
    ExpertTokenGroups,
    GroupedGemmBackendConfig,
    GroupedMmParameters,
    QuackGroupedMmBackendConfig,
    TorchGroupedMmBackendConfig,
    build_grouped_mm_backend,
)

_NUM_GROUPS = 8
_IN_FEATURES = 128
_OUT_FEATURES = 64

# fused backends zero-pad unaligned feature dims, as LoRA adapters with a small rank have
_DIMS = [
    pytest.param((_IN_FEATURES, _OUT_FEATURES), id="aligned"),
    pytest.param((5, 3), id="unaligned"),
]

_FUSED_CONFIGS = [
    pytest.param(TorchGroupedMmBackendConfig(), id="torch"),
    pytest.param(QuackGroupedMmBackendConfig(), id="quack"),
    pytest.param(GroupedGemmBackendConfig(), id="grouped_gemm"),
]


def _skip_if_missing(config: AnyGroupedMmBackendConfig) -> None:
    match config:
        case QuackGroupedMmBackendConfig():
            pytest.importorskip("quack")
        case GroupedGemmBackendConfig():
            pytest.importorskip("grouped_gemm")


def _inputs(in_features: int = _IN_FEATURES, out_features: int = _OUT_FEATURES):
    torch.manual_seed(0)
    # groups 2 and 5 are empty: their weight gradient must be zero
    counts = torch.tensor([17, 40, 0, 3, 64, 0, 1, 23])
    x = torch.randn(int(counts.sum()), in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(
        _NUM_GROUPS, in_features, out_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    grad = torch.randn(x.shape[0], out_features, device="cuda", dtype=torch.bfloat16)
    return counts, x, weight, grad


def _run(config: AnyGroupedMmBackendConfig, dims: tuple[int, int]):
    counts, x, weight, grad = _inputs(*dims)
    backend = build_grouped_mm_backend(GroupedMmParameters(in_features=dims[0], out_features=dims[1]), config)
    out = backend(x, weight, ExpertTokenGroups(counts.cuda()))
    out.backward(grad)
    return out, x.grad, weight.grad


@pytest.mark.local
@pytest.mark.parametrize("config", _FUSED_CONFIGS)
@pytest.mark.parametrize("dims", _DIMS)
def test_matches_eager(config, dims) -> None:
    _skip_if_missing(config)

    expected = _run(EagerGroupedMmBackendConfig(), dims)
    actual = _run(config, dims)

    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)
    assert actual[2][2].abs().max() == 0
    assert actual[2][5].abs().max() == 0


@pytest.mark.local
@pytest.mark.parametrize("config", _FUSED_CONFIGS)
@pytest.mark.parametrize(
    ("direction", "expect_input_grad", "expect_weight_grad"),
    [(GradDirection.inputs, True, False), (GradDirection.weight, False, True)],
)
def test_grad_directions(config, direction, expect_input_grad, expect_weight_grad) -> None:
    _skip_if_missing(config)
    counts, x, weight, grad = _inputs()

    params = GroupedMmParameters(in_features=_IN_FEATURES, out_features=_OUT_FEATURES)
    out = build_grouped_mm_backend(params, config)(x, weight, ExpertTokenGroups(counts.cuda(), host_counts=counts))
    with GLOBAL_GRAD_CONTEXT.with_directions(direction):
        out.backward(grad)

    assert (x.grad is not None) == expect_input_grad
    assert (weight.grad is not None) == expect_weight_grad
