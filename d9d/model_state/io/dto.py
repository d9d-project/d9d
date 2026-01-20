from pydantic import BaseModel


class ModelStateIndexMeta(BaseModel):
    """
    Metadata for the model state index.

    Attributes:
        total_size: Total size of the model parameters in bytes.
    """

    total_size: int


class ModelStateIndex(BaseModel):
    """
    Represents the content of the `model.safetensors.index.json` file.

    This index maps every weight name to the specific .safetensors file containing it.

    Attributes:
        metadata: Global metadata about the checkpoint.
        weight_map: Mapping from parameter name to filename.
    """

    metadata: ModelStateIndexMeta
    weight_map: dict[str, str]


MODEL_STATE_INDEX_FILE_NAME = "model.safetensors.index.json"
