from collections import defaultdict
from enum import StrEnum


class ScheduleStyle(StrEnum):
    """
    Defines the strategy for mapping logical stages to physical ranks.

    Attributes:
        loop: Assigns stages in a round-robin circular fashion (mod pp_size).
        v: Assigns stages in a zig-zag V-shape pattern. Useful for interleaved 1F1B schedules.
    """

    loop = "loop"
    v = "v"


def build_stage_to_host_rank_topology(pp_size: int, num_stages: int, style: ScheduleStyle) -> dict[int, int]:
    """
    Constructs the mapping from stage index to rank index.

    Args:
        pp_size: Number of pipeline parallel ranks.
        num_stages: Total number of model stages.
        style: The topology style to use for assignment.

    Returns:
        A dictionary mapping stage IDs to Rank IDs.

    Raises:
        ValueError: If the style is unknown or if V-style parameters are invalid
            (num_stages must be divisible by pp_size).
    """

    match style:
        case ScheduleStyle.loop:
            return {stage_index: stage_index % pp_size for stage_index in range(num_stages)}
        case ScheduleStyle.v:
            if num_stages % pp_size != 0:
                raise ValueError(
                    f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size} for V schedules"
                )

            result = {}
            rank_index = 0
            for stage_index in range(num_stages):
                result[stage_index] = rank_index
                if (stage_index + 1) % pp_size == 0:
                    continue
                if (stage_index // pp_size) % 2 == 0:
                    rank_index += 1
                else:
                    rank_index -= 1
            return result
        case _:
            raise ValueError()


def invert_stage_to_host_rank_topology(stage_to_host: dict[int, int]) -> dict[int, list[int]]:
    """
    Inverts the topology mapping to list execution stages per rank.

    Args:
        stage_to_host: Mapping from stage index to rank index.

    Returns:
        A dictionary where keys are Rank IDs and values are lists of Stage IDs
        managed by that rank.
    """

    host_to_stage = defaultdict(list)
    for stage_idx, host in stage_to_host.items():
        host_to_stage[host].append(stage_idx)
    return dict(host_to_stage)
