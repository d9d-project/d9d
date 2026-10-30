from typing import Literal

from pydantic import BaseModel, ConfigDict


class MLflowConfig(BaseModel):
    """Configuration for the MLflow tracker backend.

    Attributes:
        provider: Discriminator field. Always ``"mlflow"``.
        experiment: The name of the MLflow experiment. The tracker creates it if it does not exist.
        tracking_uri: The URI of the tracking server or store, e.g. ``"http://mlflow:5000"``. ``None`` selects the
            ``MLFLOW_TRACKING_URI`` environment variable or the default of MLflow.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["mlflow"] = "mlflow"

    experiment: str
    tracking_uri: str | None = None
