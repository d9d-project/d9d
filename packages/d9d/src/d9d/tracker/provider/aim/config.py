from typing import Literal

from pydantic import BaseModel, ConfigDict


class AimConfig(BaseModel):
    """Configuration for the Aim tracker backend.

    Attributes:
        provider: Discriminator field. Always ``"aim"``.
        repo: The path or URL of the Aim repository.
        log_system_params: Whether to log system resource usage (CPU, GPU, memory).
        capture_terminal_logs: Whether to capture stdout and stderr.
        system_tracking_interval: The interval of system monitoring in seconds.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["aim"] = "aim"

    repo: str
    log_system_params: bool = True
    capture_terminal_logs: bool = True
    system_tracking_interval: int = 10
