from unittest.mock import Mock

import pytest
from d9d.pipelining.runtime.action import (
    BackwardFullInputComputeAction,
    BackwardWeightComputeAction,
    BackwardWindowComputeAction,
    ForwardComputeAction,
    ForwardWindowComputeAction,
)
from d9d.pipelining.runtime.program_cache import merge_windows, plan_reductions


@pytest.mark.local
def test_merge_windows_joins_consecutive_computes_of_one_stage():
    program = [
        ForwardComputeAction(stage_idx=0, microbatch_idx=0),
        ForwardComputeAction(stage_idx=0, microbatch_idx=1),
        BackwardFullInputComputeAction(stage_idx=0, microbatch_idx=0, full_backward=True),
        BackwardFullInputComputeAction(stage_idx=0, microbatch_idx=1, full_backward=True),
    ]

    assert merge_windows(program) == [
        ForwardWindowComputeAction(stage_idx=0, microbatches=(0, 1)),
        BackwardWindowComputeAction(stage_idx=0, microbatches=(0, 1)),
    ]


@pytest.mark.local
def test_merge_windows_keeps_alternating_and_split_computes():
    program = [
        ForwardComputeAction(stage_idx=0, microbatch_idx=0),
        BackwardFullInputComputeAction(stage_idx=0, microbatch_idx=0, full_backward=False),
        ForwardComputeAction(stage_idx=0, microbatch_idx=1),
        BackwardWeightComputeAction(stage_idx=0, microbatch_idx=0),
    ]

    assert merge_windows(program) == program


@pytest.mark.local
def test_only_the_last_weight_writing_action_of_a_stage_is_final():
    runtime = Mock()
    runtime.reduction_plan.side_effect = lambda stage, final: (stage, final)
    program = [
        BackwardFullInputComputeAction(stage_idx=0, microbatch_idx=0, full_backward=False),
        BackwardWeightComputeAction(stage_idx=0, microbatch_idx=0),
        BackwardFullInputComputeAction(stage_idx=0, microbatch_idx=1, full_backward=True),
        BackwardWindowComputeAction(stage_idx=1, microbatches=(0, 1)),
    ]

    assert plan_reductions(program, runtime) == {
        (0, 0): (0, False),
        (0, 1): (0, True),
        (1, 1): (1, True),
    }
