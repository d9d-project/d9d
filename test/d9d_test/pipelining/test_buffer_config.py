import pytest
import torch
from d9d.pipelining.infra.schedule.component.runtime.executor import _BufferConfig  # noqa: PLC2701


@pytest.mark.local
def test_identical_packs_share_a_key():
    pack_a = ({"x": torch.empty(4, 8)}, {"x": torch.empty(4, 8)})
    pack_b = ({"x": torch.empty(4, 8)}, {"x": torch.empty(4, 8)})

    assert _BufferConfig.of(pack_a) == _BufferConfig.of(pack_b)


@pytest.mark.local
def test_different_microbatch_count_differs():
    one = ({"x": torch.empty(4, 8)},)
    two = ({"x": torch.empty(4, 8)}, {"x": torch.empty(4, 8)})

    assert _BufferConfig.of(one) != _BufferConfig.of(two)


@pytest.mark.local
def test_varying_shape_within_pack_is_captured():
    # Same microbatch count, but the second microbatch has a different shape.
    uniform = ({"x": torch.empty(4, 8)}, {"x": torch.empty(4, 8)})
    varying = ({"x": torch.empty(4, 8)}, {"x": torch.empty(6, 8)})

    assert _BufferConfig.of(uniform) != _BufferConfig.of(varying)


@pytest.mark.local
def test_dtype_difference_is_captured():
    a = ({"x": torch.empty(4, 8, dtype=torch.float32)},)
    b = ({"x": torch.empty(4, 8, dtype=torch.bfloat16)},)

    assert _BufferConfig.of(a) != _BufferConfig.of(b)


@pytest.mark.local
def test_key_is_order_insensitive_across_names():
    a = ({"x": torch.empty(4, 8), "y": torch.empty(2)},)
    b = ({"y": torch.empty(2), "x": torch.empty(4, 8)},)

    assert _BufferConfig.of(a) == _BufferConfig.of(b)
