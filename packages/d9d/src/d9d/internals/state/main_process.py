from typing import Any

from torch.distributed.checkpoint.stateful import Stateful

from d9d.core.dist_context import DistributedContext


def state_dict_main_process(dist_context: DistributedContext, obj: Stateful) -> dict[str, Any]:
    """Returns the state dict of an object on the main process only.

    Use it to checkpoint components whose state is kept by the main rank. Other ranks save nothing, so the
    checkpoint holds no duplicates.

    Args:
        dist_context: The distributed context that tells whether this is the main process.
        obj: The stateful object to save.

    Returns:
        The state of the object under the ``"main_process"`` key on the main rank, an empty dict on other ranks.
    """
    if dist_context.is_main_process:
        return {"main_process": obj.state_dict()}
    else:
        return {}


def load_state_dict_main_process(dist_context: DistributedContext, obj: Stateful, state_dict: dict[str, Any]):
    """Restores the state of an object on the main process only.

    Args:
        dist_context: The distributed context that tells whether this is the main process.
        obj: The stateful object to restore.
        state_dict: The state dict created by ``state_dict_main_process``.
    """
    if dist_context.is_main_process:
        obj.load_state_dict(state_dict["main_process"])
