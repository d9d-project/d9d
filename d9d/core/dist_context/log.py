import logging
import sys


def build_dist_logger(qualifier: str, level: int) -> logging.Logger:
    """
    Configures and returns a logger instance for d9d.

    The logger is configured to write to stdout with a formatter that includes
    the provided rank qualifier, allowing for easier debugging in distributed logs.

    Args:
        qualifier: A string identifying the current rank's position in the mesh.
        level: Log level to set by default

    Returns:
        A configured logging.Logger instance.
    """

    dist_logger = logging.getLogger('d9d')
    dist_logger.setLevel(level)
    dist_logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    formatter = logging.Formatter(
        f"[d9d] [{qualifier}] %(asctime)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    dist_logger.addHandler(ch)
    return dist_logger
