from typing import Literal

from pydantic import BaseModel, ConfigDict


class TensorBoardConfig(BaseModel):
    """Configuration for the TensorBoard tracker backend.

    Attributes:
        provider: Discriminator field. Always ``"tensorboard"``.
        log_dir: The directory of the event files of the run.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["tensorboard"] = "tensorboard"

    log_dir: str
