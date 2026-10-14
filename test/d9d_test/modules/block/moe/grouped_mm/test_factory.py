import pytest
from d9d.module.block.moe.grouped_mm import (
    EagerGroupedMmBackendConfig,
    GroupedGemmBackendConfig,
    GroupedMmParameters,
    QuackGroupedMmBackendConfig,
    TorchGroupedMmBackendConfig,
    build_grouped_mm_backend,
)
from d9d.module.block.moe.grouped_mm import factory as factory_mod

_ENV_VAR = "D9D_BACKEND_AUTO_GROUPED_MM"
_ALIGNED = GroupedMmParameters(in_features=64, out_features=32)


def _torch_cls() -> type:
    from d9d.module.block.moe.grouped_mm.impl.torch_grouped_mm import TorchGroupedMm

    return TorchGroupedMm


def _quack_cls() -> type:
    pytest.importorskip("quack")
    from d9d.module.block.moe.grouped_mm.impl.quack import QuackGroupedMm

    return QuackGroupedMm


def _grouped_gemm_cls() -> type:
    pytest.importorskip("grouped_gemm")
    from d9d.module.block.moe.grouped_mm.impl.grouped_gemm import GroupedGemm

    return GroupedGemm


def _eager_cls() -> type:
    from d9d.module.block.moe.grouped_mm.impl.eager import EagerGroupedMm

    return EagerGroupedMm


@pytest.mark.local
@pytest.mark.parametrize(
    ("config", "expected_cls"),
    [
        (TorchGroupedMmBackendConfig(), _torch_cls),
        (QuackGroupedMmBackendConfig(), _quack_cls),
        (GroupedGemmBackendConfig(), _grouped_gemm_cls),
        (EagerGroupedMmBackendConfig(), _eager_cls),
    ],
)
def test_explicit_dispatch(config, expected_cls) -> None:
    assert isinstance(build_grouped_mm_backend(_ALIGNED, config), expected_cls())


@pytest.mark.local
def test_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV_VAR, '{"kind": "eager"}')
    assert isinstance(build_grouped_mm_backend(_ALIGNED, backend_config=None), _eager_cls())


@pytest.mark.local
def test_explicit_config_wins_over_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV_VAR, '{"kind": "eager"}')
    assert isinstance(build_grouped_mm_backend(_ALIGNED, TorchGroupedMmBackendConfig()), _torch_cls())


def _patch_environment(
    monkeypatch: pytest.MonkeyPatch, capability, has_quack: bool, has_grouped_gemm: bool, has_cuda: bool = True
) -> None:
    installed = {name for name, present in (("quack", has_quack), ("grouped_gemm", has_grouped_gemm)) if present}
    monkeypatch.delenv(_ENV_VAR, raising=False)
    monkeypatch.setattr(factory_mod.torch.cuda, "is_available", lambda: has_cuda)
    monkeypatch.setattr(factory_mod.torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(factory_mod.importlib.util, "find_spec", lambda name: object() if name in installed else None)


@pytest.mark.local
@pytest.mark.parametrize(
    ("capability", "has_quack", "has_grouped_gemm", "expected_cls"),
    [
        # sm90 and newer: quack whenever it is installed
        ((9, 0), True, True, _quack_cls),
        ((10, 0), True, False, _quack_cls),
        ((12, 0), True, False, _quack_cls),
        # sm90 and sm100 without quack: fused torch grouped_mm, even with grouped_gemm installed
        ((9, 0), False, True, _torch_cls),
        ((10, 0), False, False, _torch_cls),
        # sm80: grouped_gemm has its real grouped kernels there; quack has none
        ((8, 0), True, True, _grouped_gemm_cls),
        ((8, 0), True, False, _eager_cls),
        # torch grouped_mm loops over the groups with host syncs outside sm90 and sm100
        ((11, 0), False, True, _eager_cls),
        ((12, 0), False, False, _eager_cls),
    ],
)
def test_auto_detect_priority(
    monkeypatch: pytest.MonkeyPatch, capability, has_quack, has_grouped_gemm, expected_cls
) -> None:
    expected = expected_cls()
    _patch_environment(monkeypatch, capability, has_quack=has_quack, has_grouped_gemm=has_grouped_gemm)
    assert isinstance(build_grouped_mm_backend(_ALIGNED, backend_config=None), expected)


@pytest.mark.local
def test_auto_detect_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _eager_cls()
    _patch_environment(monkeypatch, (9, 0), has_quack=True, has_grouped_gemm=True, has_cuda=False)
    assert isinstance(build_grouped_mm_backend(_ALIGNED, backend_config=None), expected)
