import logging
import sys


def build_dist_logger(qualifier: str, level: int) -> logging.Logger:
    """Configures the ``d9d`` logger to write to stdout and returns it.

    Each line includes the rank qualifier, so that ranks can be told apart in distributed logs. Existing
    handlers of the ``d9d`` logger are removed.

    Args:
        qualifier: The string that identifies the position of the current rank in the mesh.
        level: The log level.

    Returns:
        The configured ``d9d`` logger.
    """
    dist_logger = logging.getLogger("d9d")
    dist_logger.setLevel(level)
    dist_logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    formatter = logging.Formatter(f"[d9d] [{qualifier}] %(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    dist_logger.addHandler(ch)
    return dist_logger
