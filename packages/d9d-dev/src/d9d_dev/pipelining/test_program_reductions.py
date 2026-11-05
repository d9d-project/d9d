import pytest
from d9d.parallel.runtime import BucketId, ReductionPlan
from d9d.pipelining.runtime.action import (
    BackwardFullInputComputeAction,
    BackwardWeightComputeAction,
    BackwardWindowComputeAction,
    ComposeAction,
    ForwardComputeAction,
    ForwardWindowComputeAction,
)
from d9d.pipelining.runtime.program_cache import merge_windows, plan_reductions


class _OneBucketRuntime:
    """A runtime with one bucket per stage, which reduces after the first segment."""

    def reduction_plan(self, stage_index, final):
        return ReductionPlan(final, {0: (BucketId(stage_index, 0),)})


def _stages_and_finals(plans):
    return {key: (plan.after_segment[0][0].stage, plan.final) for key, plan in plans.items()}


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
    program = [
        BackwardFullInputComputeAction(stage_idx=0, microbatch_idx=0, full_backward=False),
        BackwardWeightComputeAction(stage_idx=0, microbatch_idx=0),
        BackwardFullInputComputeAction(stage_idx=0, microbatch_idx=1, full_backward=True),
        BackwardWindowComputeAction(stage_idx=1, microbatches=(0, 1)),
    ]

    assert _stages_and_finals(plan_reductions(program, _OneBucketRuntime())) == {
        (0, 0): (0, False),
        (0, 1): (0, True),
        (1, 1): (1, True),
    }


@pytest.mark.local
def test_reductions_cover_nested_compose_actions():
    backward = BackwardFullInputComputeAction(stage_idx=0, microbatch_idx=0, full_backward=True)
    program = [ComposeAction(actions=(ComposeAction(actions=(backward,)),))]

    assert _stages_and_finals(plan_reductions(program, _OneBucketRuntime())) == {(0, 0): (0, True)}
