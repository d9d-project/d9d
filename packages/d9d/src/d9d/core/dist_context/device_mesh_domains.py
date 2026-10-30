import abc
from typing import TYPE_CHECKING

from torch.distributed import DeviceMesh, init_device_mesh

if TYPE_CHECKING:
    from .params import DeviceMeshParameters


class DeviceMeshDomain(abc.ABC):
    """Abstract base class for a domain that builds one device mesh.

    A domain arranges the GPUs into a multidimensional mesh that serves specific parallelism techniques.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The unique name of this domain."""
        ...

    @abc.abstractmethod
    def build_mesh(self, params: "DeviceMeshParameters") -> DeviceMesh:
        """Builds the device mesh of this domain.

        Args:
            params: The parallelism degrees.

        Returns:
            The device mesh of this domain.
        """
        ...


REGULAR_DOMAIN = "regular"


class RegularDomain(DeviceMeshDomain):
    """Domain that builds the most granular mesh, with every parallelism in its own dimension.

    Dimensions: ``("pp", "dp_replicate", "dp_shard", "cp_shard", "cp_replicate", "tp")``.
    """

    @property
    def name(self) -> str:
        return "regular"

    def build_mesh(self, params: "DeviceMeshParameters") -> DeviceMesh:
        return init_device_mesh(
            device_type="cuda",
            mesh_shape=(
                params.pipeline_parallel,
                params.data_parallel_replicate,
                params.data_parallel_shard,
                params.context_parallel_shard,
                params.context_parallel_replicate,
                params.tensor_parallel,
            ),
            mesh_dim_names=(
                "pp",
                "dp_replicate",
                "dp_shard",
                "cp_shard",
                "cp_replicate",
                "tp",
            ),
        )


EXPERT_DOMAIN = "expert"


class ExpertDomain(DeviceMeshDomain):
    """Domain that builds the mesh for Mixture-of-Experts (MoE) layers.

    Dimensions: ``("pp", "ep_replicate", "ep_shard")``.
    """

    @property
    def name(self) -> str:
        return EXPERT_DOMAIN

    def build_mesh(self, params: "DeviceMeshParameters") -> DeviceMesh:
        replicate_degree = (
            params.data_parallel_replicate
            * params.context_parallel_replicate
            * params.data_parallel_shard
            * params.context_parallel_shard
        )
        return init_device_mesh(
            device_type="cuda",
            mesh_shape=(
                params.pipeline_parallel,
                replicate_degree // params.expert_parallel,
                params.expert_parallel,
            ),
            mesh_dim_names=(
                "pp",
                "ep_replicate",
                "ep_shard",
            ),
        )


DENSE_DOMAIN = "dense"


class DenseDomain(DeviceMeshDomain):
    """Domain that builds the mesh for dense layers.

    Dimensions: ``("pp", "dp_replicate", "dp_cp_shard", "cp_replicate", "tp")``.
    """

    @property
    def name(self) -> str:
        return DENSE_DOMAIN

    def build_mesh(self, params: "DeviceMeshParameters") -> DeviceMesh:
        return init_device_mesh(
            device_type="cuda",
            mesh_shape=(
                params.pipeline_parallel,
                params.data_parallel_replicate,
                params.data_parallel_shard * params.context_parallel_shard,
                params.context_parallel_replicate,
                params.tensor_parallel,
            ),
            mesh_dim_names=(
                "pp",
                "dp_replicate",
                "dp_cp_shard",
                "cp_replicate",
                "tp",
            ),
        )


BATCH_DOMAIN = "batch"


class BatchDomain(DeviceMeshDomain):
    """Domain that builds the mesh for distributing input data.

    Dimensions: ``("pp", "dp", "cp", "tp")``.
    """

    @property
    def name(self) -> str:
        return BATCH_DOMAIN

    def build_mesh(self, params: "DeviceMeshParameters") -> DeviceMesh:
        return init_device_mesh(
            device_type="cuda",
            mesh_shape=(
                params.pipeline_parallel,
                params.data_parallel_replicate * params.data_parallel_shard,
                params.context_parallel_replicate * params.context_parallel_shard,
                params.tensor_parallel,
            ),
            mesh_dim_names=(
                "pp",
                "dp",
                "cp",
                "tp",
            ),
        )


FLAT_DOMAIN = "flat"


class FlatDomain(DeviceMeshDomain):
    """Domain that builds a single-dimension mesh with all the processes.

    Dimensions: ``("world",)``.
    """

    @property
    def name(self) -> str:
        return FLAT_DOMAIN

    def build_mesh(self, params: "DeviceMeshParameters") -> DeviceMesh:
        mesh_shape = (
            params.pipeline_parallel
            * params.data_parallel_replicate
            * params.data_parallel_shard
            * params.context_parallel_replicate
            * params.context_parallel_shard
            * params.tensor_parallel
        )
        return init_device_mesh(
            device_type="cuda",
            mesh_shape=(mesh_shape,),
            mesh_dim_names=("world",),
        )


ALL_DOMAIN_PROVIDERS: list[DeviceMeshDomain] = [
    RegularDomain(),
    DenseDomain(),
    ExpertDomain(),
    BatchDomain(),
    FlatDomain(),
]
