import abc
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

from .params import DeviceMeshParameters

DimT = TypeVar("DimT", bound=StrEnum)


@dataclass(frozen=True)
class MeshDim(Generic[DimT]):
    """One dimension of a device mesh.

    Attributes:
        name: The name of the dimension.
        size: The number of ranks along the dimension.
    """

    name: DimT
    size: int


class DeviceMeshDomain(abc.ABC, Generic[DimT]):
    """Abstract base class for a domain: one arrangement of all ranks into a device mesh."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of this domain, unique within a context."""

    @abc.abstractmethod
    def mesh_dims(self, params: DeviceMeshParameters) -> tuple[MeshDim[DimT], ...]:
        """Returns the dimensions of the mesh of this domain, outermost first.

        The context lays out the ranks along them row-major, as ``init_device_mesh()`` does. So a dimension that
        keeps the order of the dimensions of ``BASE_DOMAIN`` groups the ranks as they do.

        Args:
            params: The parallelism degrees and the sections of the built-in domains.

        Returns:
            The dimensions of the mesh.
        """
