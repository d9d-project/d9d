import copy
import dataclasses

from ..runtime.action import (
    ActionBase,
    BackwardFullInputComputeAction,
    BackwardReceiveAction,
    BackwardSendAction,
    ComposeAction,
    ForwardComputeAction,
    ForwardReceiveAction,
    ForwardSendAction,
)


def _get_sub_actions(action: ActionBase) -> tuple[ActionBase, ...]:
    if isinstance(action, ComposeAction):
        return action.actions
    return (action,)


def _check_action_communication_dependencies_fulfilled(
    action: ActionBase, rank_events: set[ActionBase], num_stages: int
) -> bool:
    match action:
        case ForwardComputeAction():
            if action.stage_idx == 0:
                return True
            if ForwardReceiveAction(action.stage_idx, action.microbatch_idx) in rank_events:
                return True
            if ForwardComputeAction(action.stage_idx - 1, action.microbatch_idx) in rank_events:
                return True
            return False
        case BackwardFullInputComputeAction():
            if action.stage_idx == num_stages - 1:
                return True
            if BackwardReceiveAction(action.stage_idx, action.microbatch_idx) in rank_events:
                return True

            next_full = BackwardFullInputComputeAction(action.stage_idx + 1, action.microbatch_idx, full_backward=True)
            next_inp = BackwardFullInputComputeAction(action.stage_idx + 1, action.microbatch_idx, full_backward=False)

            if next_full in rank_events or next_inp in rank_events:
                return True
            return False
        case _:
            return True


def check_action_communication_dependencies_fulfilled(
    action: ActionBase, rank_events: set[ActionBase], num_stages: int
) -> bool:
    """Checks whether the data dependencies of an action are met.

    A compute action can run when its inputs are available in ``rank_events``. Inputs are available
    if a previous stage computed them on this rank or if they were received from a remote rank.

    Args:
        action: The action to check.
        rank_events: A set of actions already completed on this rank.
        num_stages: Total number of stages in the pipeline.

    Returns:
        ``True`` if all dependencies are met, ``False`` otherwise.
    """
    return all(
        _check_action_communication_dependencies_fulfilled(sub, rank_events, num_stages)
        for sub in _get_sub_actions(action)
    )


@dataclasses.dataclass(kw_only=True)
class _CommunicationPackage:
    send: ActionBase
    recv: ActionBase
    sends_to_rank: int


def _create_communications_for_action(
    action: ActionBase,
    num_stages: int,
    stage_to_rank: dict[int, int],
) -> _CommunicationPackage | None:
    match action:
        case ForwardComputeAction():
            if action.stage_idx == num_stages - 1:
                return None

            curr_rank, next_rank = stage_to_rank[action.stage_idx], stage_to_rank[action.stage_idx + 1]
            if curr_rank == next_rank:
                return None

            return _CommunicationPackage(
                send=ForwardSendAction(action.stage_idx, action.microbatch_idx),
                recv=ForwardReceiveAction(action.stage_idx + 1, action.microbatch_idx),
                sends_to_rank=next_rank,
            )
        case BackwardFullInputComputeAction():
            if action.stage_idx == 0:
                return None

            curr_rank, prev_rank = stage_to_rank[action.stage_idx], stage_to_rank[action.stage_idx - 1]
            if curr_rank == prev_rank:
                return None

            return _CommunicationPackage(
                send=BackwardSendAction(action.stage_idx, action.microbatch_idx),
                recv=BackwardReceiveAction(action.stage_idx - 1, action.microbatch_idx),
                sends_to_rank=prev_rank,
            )
        case _:
            return None


def add_communication_ops(
    compute_actions: dict[int, list[ActionBase]],
    stage_to_rank: dict[int, int],
    num_stages: int,
) -> dict[int, list[ActionBase]]:
    """Injects communication actions into a computation-only schedule.

    This function simulates the compute schedule. When a compute action produces a result that
    another rank needs, it adds a send and receive pair. Each receive comes before the computes
    that depend on it, so the schedule does not deadlock.

    Args:
        compute_actions: Initial schedule containing only compute operations.
        stage_to_rank: Mapping from stage index to rank index.
        num_stages: Total number of pipeline stages.

    Returns:
        A new schedule dictionary including both compute and communication actions.

    Raises:
        RuntimeError: If the schedule simulation deadlocks.
    """
    compute_actions = copy.deepcopy(compute_actions)

    full_actions: dict[int, list[ActionBase]] = {rank: [] for rank in compute_actions}
    completed_events: dict[int, set[ActionBase]] = {rank: set() for rank in compute_actions}

    while compute_actions:
        progress = False

        for rank in sorted(compute_actions.keys()):
            if not compute_actions[rank]:
                del compute_actions[rank]
                continue

            current_action = compute_actions[rank][0]
            sub_actions = _get_sub_actions(current_action)

            if not check_action_communication_dependencies_fulfilled(
                current_action, completed_events[rank], num_stages
            ):
                continue

            full_actions[rank].append(current_action)
            compute_actions[rank].pop(0)
            progress = True

            for sub_action in sub_actions:
                completed_events[rank].add(sub_action)

                comm_pkg = _create_communications_for_action(
                    sub_action, num_stages=num_stages, stage_to_rank=stage_to_rank
                )
                if comm_pkg:
                    full_actions[rank].append(comm_pkg.send)
                    completed_events[rank].add(comm_pkg.send)

                    # Adding the receive to the peer rank unblocks its dependent compute.
                    full_actions[comm_pkg.sends_to_rank].append(comm_pkg.recv)
                    completed_events[comm_pkg.sends_to_rank].add(comm_pkg.recv)

        if not progress and compute_actions:
            raise RuntimeError(
                "The schedule simulation deadlocked: no rank can run its next action. "
                "Check the compute order produced by the program builder."
            )

    return full_actions
