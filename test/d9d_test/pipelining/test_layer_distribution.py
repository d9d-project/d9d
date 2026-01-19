import pytest

from d9d.pipelining.api import PipelineStageInfo, distribute_layers_for_pipeline_stage


@pytest.mark.parametrize(
    "num_layers,num_stages,pre,post,expected_distribution",
    [
        # Standard, symmetric case
        # 24 layers, 4 stages, 1 pre, 1 post.
        # Virtual layers = 24 + 1 + 1 = 26. 26 / 4 = 6 rem 2.
        # Virtual distribution: [7, 7, 6, 6]. Adjustments: [-1, 0, 0, -1].
        # Final counts: [6, 7, 6, 5].
        (24, 4, 1, 1, [(0, 6), (6, 13), (13, 19), (19, 24)]),

        # Asymmetric load
        # 30 layers, 4 stages, 0 pre, 2 post.
        # Virtual layers = 30 + 0 + 2 = 32. 32 / 4 = 8 rem 0.
        # Virtual distribution: [8, 8, 8, 8]. Adjustments: [0, 0, 0, -2].
        # Final counts: [8, 8, 8, 6].
        (30, 4, 0, 2, [(0, 8), (8, 16), (16, 24), (24, 30)]),

        # Edge case - Single stage pipeline.
        # It is both first and last, so both adjustments apply.
        # Virtual: 10 + 1 + 2 = 13. Virtual dist: [13]. Adjustment: -(1+2)=-3. Final: 10.
        (10, 1, 1, 2, [(0, 10)]),

        # Edge case - Two stage pipeline.
        # Stage 0 is first, Stage 1 is last.
        # Virtual: 12 + 1 + 1 = 14. 14 / 2 = 7 rem 0.
        # Virtual dist: [7, 7]. Adjustments: [-1, -1]. Final: [6, 6].
        (12, 2, 1, 1, [(0, 6), (6, 12)]),

        # Perfectly divisible, no virtual layers. Should be even.
        (32, 4, 0, 0, [(0, 8), (8, 16), (16, 24), (24, 32)]),

        # Not perfectly divisible, no virtual layers. Remainder goes to early stages.
        (34, 4, 0, 0, [(0, 9), (9, 18), (18, 26), (26, 34)]),  # Counts: [9, 9, 8, 8]
    ]
)
@pytest.mark.local
def test_distribute_layers_scenarios(
        num_layers: int, num_stages: int, pre: int, post: int, expected_distribution: list[tuple[int, int]]
):
    all_layers_accounted_for = 0
    for i in range(num_stages):
        stage_info = PipelineStageInfo(current_stage=i, num_stages=num_stages)

        start, end = distribute_layers_for_pipeline_stage(
            num_layers,
            pre,
            post,
            stage_info
        )

        expected_start, expected_end = expected_distribution[i]

        assert start == expected_start, f"Stage {i} start index mismatch"
        assert end == expected_end, f"Stage {i} end index mismatch"
        all_layers_accounted_for += (end - start)


@pytest.mark.parametrize(
    "num_layers,num_stages,pre,post",
    [
        # Zero layers to distribute.
        (0, 4, 1, 1),
        # Edge case - Few layers, many stages.
        # Some stages should get zero layers.
        (2, 4, 1, 1)
    ]
)
@pytest.mark.local
def test_distribute_layers_invalid_input(num_layers, num_stages, pre, post):
    """
    Tests that the function raises ValueError for invalid stage configurations.
    """
    for stage in range(num_stages):
        with pytest.raises(ValueError):
            stage_info = PipelineStageInfo(current_stage=stage, num_stages=num_stages)
            distribute_layers_for_pipeline_stage(
                num_layers=num_layers,
                num_virtual_layers_pre=pre,
                num_virtual_layers_post=post,
                stage=stage_info
            )
