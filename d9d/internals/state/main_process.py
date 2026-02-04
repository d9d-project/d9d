from typing import Any

from torch.distributed.checkpoint.stateful import Stateful

from d9d.core.dist_context import DistributedContext


def state_dict_main_process(dist_context: DistributedContext, obj: Stateful) -> dict[str, Any]:
    """
    Retrieves the state dictionary of an object only on the main process.

    This is useful for checkpointing components that track global state primarily
    managed by the driver/main rank, ensuring that non-main ranks return an empty
    state to avoid duplication or synchronization issues during checkpointing.

    Args:
        dist_context: The distributed context to check for main process status.
        obj: The stateful object to serialize.

    Returns:
        A dictionary containing the object's state under the 'main_process' key on
            the main rank, and an empty dictionary on all other ranks.
    """

    if dist_context.is_main_process:
        return {
            "main_process": obj.state_dict()
        }
    else:
        return {}


def load_state_dict_main_process(dist_context: DistributedContext, obj: Stateful, state_dict: dict[str, Any]):
    """
    Restores the state dictionary of an object only on the main process.

    Args:
        dist_context: The distributed context to check for main process status.
        obj: The stateful object to restore.
        state_dict: The state dictionary created by "state_dict_main_process" function.
    """

    if dist_context.is_main_process:
        obj.load_state_dict(state_dict["main_process"])
