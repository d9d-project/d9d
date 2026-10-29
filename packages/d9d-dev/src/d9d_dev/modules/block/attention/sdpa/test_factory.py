import pytest
from d9d.module.block.attention.sdpa import (
    EagerSdpaBackendConfig,
    FlashAttention2SdpaBackendConfig,
    FlashAttention4SdpaBackendConfig,
    SdpaParameters,
    TorchSdpaBackendConfig,
    build_sdpa_backend,
)
from d9d.module.block.attention.sdpa import factory as factory_mod

_ENV_VAR = "D9D_BACKEND_AUTO_SDPA"


def _patch_specs(monkeypatch: pytest.MonkeyPatch, available: set[str]) -> None:
    monkeypatch.setattr(
        factory_mod.importlib.util,
        "find_spec",
        lambda name: object() if name in available else None,
    )


def _eager_cls() -> type:
    from d9d.module.block.attention.sdpa.impl.eager import EagerSdpa

    return EagerSdpa


def _torch_cls() -> type:
    from d9d.module.block.attention.sdpa.impl.torch_sdpa import TorchSdpa

    return TorchSdpa


def _flash2_cls() -> type:
    pytest.importorskip("flash_attn")
    from d9d.module.block.attention.sdpa.impl.flash2 import FlashAttention2Sdpa

    return FlashAttention2Sdpa


def _flash4_cls() -> type:
    pytest.importorskip("flash_attn.cute")
    from d9d.module.block.attention.sdpa.impl.flash4 import FlashAttention4Sdpa

    return FlashAttention4Sdpa


@pytest.mark.local
@pytest.mark.parametrize(
    ("config", "expected_cls"),
    [
        (EagerSdpaBackendConfig(), _eager_cls),
        (TorchSdpaBackendConfig(), _torch_cls),
        (FlashAttention2SdpaBackendConfig(), _flash2_cls),
        (FlashAttention4SdpaBackendConfig(), _flash4_cls),
    ],
)
def test_explicit_dispatch(config, expected_cls) -> None:
    backend = build_sdpa_backend(SdpaParameters(num_sinks=None), config)
    assert isinstance(backend, expected_cls())


@pytest.mark.local
@pytest.mark.parametrize(
    ("payload", "expected_cls"),
    [
        ('{"kind": "eager"}', _eager_cls),
        ('{"kind": "torch"}', _torch_cls),
        ('{"kind": "torch", "backends": ["MATH"]}', _torch_cls),
        ('{"kind": "flash_attention_2"}', _flash2_cls),
        ('{"kind": "flash_attention_4"}', _flash4_cls),
    ],
)
def test_env_var_override(monkeypatch: pytest.MonkeyPatch, payload: str, expected_cls) -> None:
    # With no explicit config, auto-detection reads the env var and builds it.
    monkeypatch.setenv(_ENV_VAR, payload)
    backend = build_sdpa_backend(SdpaParameters(num_sinks=None), backend_config=None)
    assert isinstance(backend, expected_cls())


@pytest.mark.local
@pytest.mark.parametrize(
    ("available", "num_sinks", "window_size", "needs_attention_mask", "expected_cls"),
    [
        # Flash 4 wins whenever its kernel is present.
        ({"flash_attn", "flash_attn.cute"}, 4, (8, 0), False, _flash4_cls),
        # Flash 2 handles windows but not sinks.
        ({"flash_attn"}, None, (8, 0), False, _flash2_cls),
        # Plain torch when nothing special is requested and no flash is present.
        (set(), None, (None, None), False, _torch_cls),
        # Eager fallback: sinks requested but only flash 2 (no sink support) is present.
        ({"flash_attn"}, 4, (None, None), False, _eager_cls),
        # Eager fallback: window requested with no flash kernel at all.
        (set(), None, (8, 0), False, _eager_cls),
        # Explicit masks disqualify Flash 4/2 even when their kernels exist: torch wins.
        ({"flash_attn", "flash_attn.cute"}, None, (None, None), True, _torch_cls),
        # Explicit masks + sinks: only eager can satisfy both.
        ({"flash_attn", "flash_attn.cute"}, 4, (None, None), True, _eager_cls),
        # Explicit masks + window: torch cannot do windows, so eager is the fallback.
        ({"flash_attn", "flash_attn.cute"}, None, (8, 0), True, _eager_cls),
    ],
)
def test_auto_detect_priority(
    monkeypatch: pytest.MonkeyPatch, available, num_sinks, window_size, needs_attention_mask, expected_cls
) -> None:
    # Auto-detection only probes flash module availability, which we fake here;
    # the chosen backend is then built through the public factory.
    monkeypatch.delenv(_ENV_VAR, raising=False)
    expected = expected_cls()
    _patch_specs(monkeypatch, available)
    backend = build_sdpa_backend(
        SdpaParameters(
            num_sinks=num_sinks,
            window_size=window_size,
            needs_attention_mask=needs_attention_mask,
        ),
        backend_config=None,
    )
    assert isinstance(backend, expected)
