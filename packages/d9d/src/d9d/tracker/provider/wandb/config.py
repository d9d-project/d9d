from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WandbConfig(BaseModel):
    """Configuration for the W&B tracker backend.

    Attributes:
        provider: Discriminator field. Always ``"wandb"``.
        project: The W&B project.
        entity: The team or user that owns the project. ``None`` selects the default entity of the API key.
        group: The group of the run, e.g. to show the runs of one experiment together.
        tags: The tags of the run.
        mode: ``"online"`` sends the values to the server during training. ``"offline"`` writes them to local files,
            which ``wandb sync`` uploads later.
        dir: The directory for the local files of W&B. ``None`` selects the working directory.
        capture_terminal_logs: Whether to capture stdout and stderr.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["wandb"] = "wandb"

    project: str
    entity: str | None = None
    group: str | None = None
    tags: list[str] = Field(default_factory=list)
    mode: Literal["online", "offline"] = "online"
    dir: str | None = None
    capture_terminal_logs: bool = False
