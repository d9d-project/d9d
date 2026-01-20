from typing import Literal

from pydantic import BaseModel


class AimConfig(BaseModel):
    """
    Configuration for the Aim tracker backend.

    Attributes:
        provider: Discriminator field, must be 'aim'.
        repo: Path to the Aim repository directory or URL.
        log_system_params: Whether to log system resource usage (CPU/GPU/Memory).
        capture_terminal_logs: Whether to capture stdout/stderr.
        system_tracking_interval: Interval in seconds for system monitoring.
    """

    provider: Literal["aim"] = "aim"

    repo: str
    log_system_params: bool = True
    capture_terminal_logs: bool = True
    system_tracking_interval: int = 10
