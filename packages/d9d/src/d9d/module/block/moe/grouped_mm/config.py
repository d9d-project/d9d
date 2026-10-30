import dataclasses
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class TorchGroupedMmBackendConfig(BaseModel):
    """Configuration for the PyTorch ``torch.nn.functional.grouped_mm`` backend.

    Reads the expert offsets on the device, so it does not need the per-expert counts on the host. Fast only for bf16
    on sm90 and sm100.

    Attributes:
        kind: Discriminator field. Always ``"torch"``.
    """

    kind: Literal["torch"] = "torch"


class QuackGroupedMmBackendConfig(BaseModel):
    """Configuration for the QuACK (CuTe-DSL) variable-length GEMM backend.

    Reads the expert offsets on the device, so it does not need the per-expert counts on the host. Kernels are
    JIT-compiled on first use. Requires the ``d9d[backend-grouped-mm-quack]`` extra.

    Attributes:
        kind: Discriminator field. Always ``"quack"``.
    """

    kind: Literal["quack"] = "quack"


class GroupedGemmBackendConfig(BaseModel):
    """Configuration for the ``grouped_gemm`` library backend.

    Needs the per-expert counts on the host; getting them from the device synchronizes with it. Requires the
    ``d9d[backend-grouped-mm-grouped-gemm]`` extra.

    Attributes:
        kind: Discriminator field. Always ``"grouped_gemm"``.
    """

    kind: Literal["grouped_gemm"] = "grouped_gemm"


class EagerGroupedMmBackendConfig(BaseModel):
    """Configuration for the eager backend: one ``matmul`` per expert.

    A portable, dependency-free reference. Needs the per-expert counts on the host.

    Attributes:
        kind: Discriminator field. Always ``"eager"``.
    """

    kind: Literal["eager"] = "eager"


AnyGroupedMmBackendConfig = Annotated[
    TorchGroupedMmBackendConfig | QuackGroupedMmBackendConfig | GroupedGemmBackendConfig | EagerGroupedMmBackendConfig,
    Field(discriminator="kind"),
]


@dataclasses.dataclass(kw_only=True)
class GroupedMmParameters:
    """Structural parameters of a grouped linear layer that a grouped GEMM backend must support.

    Attributes:
        in_features: Size of the contracted dimension.
        out_features: Size of the output dimension.
    """

    in_features: int
    out_features: int
