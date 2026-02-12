import pytest
import torch
from d9d.module.block.hidden_states_aggregator import HiddenStatesAggregationMode, create_hidden_states_aggregator
from d9d.module.block.hidden_states_aggregator.mean import HiddenStatesAggregatorMean
from d9d.module.block.hidden_states_aggregator.noop import HiddenStatesAggregatorNoOp


@pytest.mark.local
@pytest.mark.parametrize(
    ("hidden_states_data", "mask_data", "expected_data"),
    [
        # Case 1: Simple 2x2x2 batch, full mask
        # Input: (Batch=2, Seq=2, Hidden=2)
        (
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[1.0, 1.0], [1.0, 1.0]],
            # Expected Mean:
            # B0: (1+3)/2, (2+4)/2 -> [2, 3]
            # B1: (5+7)/2, (6+8)/2 -> [6, 7]
            [[2.0, 3.0], [6.0, 7.0]],
        ),
        # Case 2: Masking logic
        # Input: (Batch=2, Seq=2, Hidden=1)
        (
            [[[10.0], [99.0]], [[20.0], [40.0]]],
            [[1.0, 0.0], [1.0, 1.0]],
            # Expected Mean:
            # B0: Keep index 0 only -> [10]
            # B1: Keep both -> (20+40)/2 -> [30]
            [[10.0], [30.0]],
        ),
        # Case 3: Uneven masking (3 tokens, mask middle)
        # Input: (Batch=1, Seq=3, Hidden=1)
        (
            [[[10.0], [500.0], [20.0]]],
            [[1.0, 0.0, 1.0]],
            # Expected: (10 + 20) / 2 -> 15
            [[15.0]],
        ),
    ],
)
def test_mean_aggregation_correctness(hidden_states_data, mask_data, expected_data):
    """
    Verifies that HiddenStatesAggregatorMean correctly computes masked averages
    using specific input/output examples via the public API.
    """
    hidden_states = torch.tensor(hidden_states_data)
    mask = torch.tensor(mask_data)
    expected = torch.tensor(expected_data)

    aggregator = HiddenStatesAggregatorMean(agg_mask=mask)
    aggregator.add_hidden_states(hidden_states)

    # Pack returns (Num_Adds, Batch, Hidden). We added once, so index 0.
    result = aggregator.pack_with_snapshot(None)

    assert result is not None
    assert result.shape[0] == 1

    # Check values
    assert torch.allclose(result[0], expected)


@pytest.mark.local
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [2, 4])
@pytest.mark.parametrize("hidden_dim", [4])
@pytest.mark.parametrize("num_adds", [1, 3])
@pytest.mark.parametrize("snapshot_len", [0, 2])
def test_mean_aggregator_lifecycle(batch_size, seq_len, hidden_dim, num_adds, snapshot_len):
    mask = torch.ones(batch_size, seq_len)
    aggregator = HiddenStatesAggregatorMean(agg_mask=mask)

    inputs = []
    for _ in range(num_adds):
        t = torch.randn(batch_size, seq_len, hidden_dim)
        inputs.append(t)
        aggregator.add_hidden_states(t)

    if snapshot_len > 0:
        snapshot = torch.randn(snapshot_len, batch_size, hidden_dim)
    else:
        snapshot = None

    result = aggregator.pack_with_snapshot(snapshot)

    expected_len = snapshot_len + num_adds
    assert result is not None
    assert result.shape == (expected_len, batch_size, hidden_dim)

    if snapshot_len > 0:
        assert torch.equal(result[:snapshot_len], snapshot)

    accumulated_result = result[snapshot_len:]
    for i, t_in in enumerate(inputs):
        expected_mean = t_in.mean(dim=1)  # Simple mean because mask is all ones
        assert torch.allclose(accumulated_result[i], expected_mean)

    res_cleared = aggregator.pack_with_snapshot(snapshot)
    assert res_cleared is None


@pytest.mark.local
def test_noop_aggregator():
    agg = HiddenStatesAggregatorNoOp()

    t = torch.randn(10, 10, 10)
    agg.add_hidden_states(t)

    res = agg.pack_with_snapshot(None)
    assert res is None

    res_snap = agg.pack_with_snapshot(torch.randn(1, 1))
    assert res_snap is None


@pytest.mark.local
@pytest.mark.parametrize(
    ("mode", "provide_mask", "expect_class", "expect_error"),
    [
        (HiddenStatesAggregationMode.no, False, HiddenStatesAggregatorNoOp, None),
        (HiddenStatesAggregationMode.no, True, HiddenStatesAggregatorNoOp, None),
        (HiddenStatesAggregationMode.mean, True, HiddenStatesAggregatorMean, None),
        (HiddenStatesAggregationMode.mean, False, None, ValueError),
    ],
)
def test_factory_creation(mode, provide_mask, expect_class, expect_error):
    mask = torch.ones(1, 1) if provide_mask else None

    if expect_error:
        with pytest.raises(expect_error, match="aggregation mask"):
            create_hidden_states_aggregator(mode, mask)
    else:
        agg = create_hidden_states_aggregator(mode, mask)
        assert isinstance(agg, expect_class)
