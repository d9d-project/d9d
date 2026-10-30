from typing import Any

from d9d.core import pytree
from d9d.tracker import TrackerPoint


def series_key(point: TrackerPoint) -> str:
    """Builds the key of a series for a tracker that has no tags.

    Args:
        point: The point to build the key for.

    Returns:
        The values of the context in their order, then the name, joined with ``/``. For example, the context
        ``{"stage": "train"}`` and the name ``loss`` give ``train/loss``.
    """
    return "/".join([*point.context.values(), point.name])


def flatten_hparams(hparams: dict[str, Any]) -> dict[str, Any]:
    """Flattens nested hyperparameters for a tracker that takes only flat key-value pairs.

    Args:
        hparams: The hyperparameters.

    Returns:
        The leaves of ``hparams``, keyed by their path joined with ``/``, e.g. ``run/lr``.
    """
    # Keep None values: by default, the PyTree traversal drops them as empty nodes.
    leaves = pytree.tree_leaves_with_path(hparams, is_leaf=lambda node: node is None)
    return {"/".join(str(segment) for segment in path): value for path, value in leaves}
