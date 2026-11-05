import logging
import sys

from .builtin import BASE_DOMAIN, BaseDim
from .configured import DistributedContext


def build_dist_logger(dist_context: DistributedContext, level: int) -> logging.Logger:
    """Configures the ``d9d`` logger to write to stdout and returns it.

    Each line starts with the coordinates of this rank in ``BASE_DOMAIN``, e.g. ``[d9d] [pp:0-dp:3-cp:0-tp:0]``, so
    that ranks can be told apart in distributed logs. Existing handlers of the ``d9d`` logger are removed.

    Args:
        dist_context: The distributed context of this process.
        level: The log level.

    Returns:
        The configured ``d9d`` logger.
    """
    qualifier = "-".join(f"{dim}:{dist_context.coordinate(BASE_DOMAIN, dim)}" for dim in BaseDim)
    dist_logger = logging.getLogger("d9d")
    dist_logger.setLevel(level)
    dist_logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    formatter = logging.Formatter(f"[d9d] [{qualifier}] %(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    dist_logger.addHandler(ch)
    return dist_logger
