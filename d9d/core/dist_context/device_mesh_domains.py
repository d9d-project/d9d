import abc
from typing import TYPE_CHECKING

from torch.distributed import DeviceMesh, init_device_mesh

if TYPE_CHECKING:
    from .params import DeviceMeshParameters


class DeviceMeshDomain(abc.ABC):
    """
    Abstract base class for a Device Mesh provider.

    A Domain defines a specific strategy for organizing available GPUs into a
    multidimensional grid (Mesh) to support specific parallelism techniques.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Returns the unique identifier for this mesh domain."""

        ...

    @abc.abstractmethod
    def build_mesh(self, params: 'DeviceMeshParameters') -> DeviceMesh:
        """
        Constructs the device mesh configuration.

        Args:
            params: Global configuration parameters for the distributed environment.

        Returns:
            The initialized PyTorch DeviceMesh for this specific domain.
        """

        ...


REGULAR_DOMAIN = 'regular'


class RegularDomain(DeviceMeshDomain):
    @property
    def name(self) -> str:
        return 'regular'

    def build_mesh(self, params: 'DeviceMeshParameters') -> DeviceMesh:
        return init_device_mesh(
            device_type='cuda',
            mesh_shape=(
                params.pipeline_parallel,
                params.data_parallel_replicate,
                params.data_parallel_shard,
                params.context_parallel_shard,
                params.context_parallel_replicate,
                params.tensor_parallel
            ),
            mesh_dim_names=(
                'pp',
                'dp_replicate',
                'dp_shard',
                'cp_shard',
                'cp_replicate',
                'tp'
            )
        )


EXPERT_DOMAIN = 'expert'


class ExpertDomain(DeviceMeshDomain):
    @property
    def name(self) -> str:
        return EXPERT_DOMAIN

    def build_mesh(self, params: 'DeviceMeshParameters') -> DeviceMesh:
        replicate_degree = (
            params.data_parallel_replicate *
            params.context_parallel_replicate *
            params.data_parallel_shard *
            params.context_parallel_shard
        )
        return init_device_mesh(
            device_type='cuda',
            mesh_shape=(
                params.pipeline_parallel,
                replicate_degree,
                params.expert_parallel
            ),
            mesh_dim_names=(
                'pp',
                'replicate',
                'ep'
            )
        )


DENSE_DOMAIN = 'dense'


class DenseDomain(DeviceMeshDomain):
    @property
    def name(self) -> str:
        return DENSE_DOMAIN

    def build_mesh(self, params: 'DeviceMeshParameters') -> DeviceMesh:
        return init_device_mesh(
            device_type='cuda',
            mesh_shape=(
                params.pipeline_parallel,
                params.data_parallel_replicate,
                params.data_parallel_shard * params.context_parallel_shard,
                params.context_parallel_replicate,
                params.tensor_parallel
            ),
            mesh_dim_names=(
                'pp',
                'dp_replicate',
                'dp_cp_shard',
                'cp_replicate',
                'tp'
            )
        )


BATCH_DOMAIN = 'batch'


class BatchDomain(DeviceMeshDomain):
    @property
    def name(self) -> str:
        return BATCH_DOMAIN

    def build_mesh(self, params: 'DeviceMeshParameters') -> DeviceMesh:
        return init_device_mesh(
            device_type='cuda',
            mesh_shape=(
                params.pipeline_parallel,
                params.data_parallel_replicate * params.data_parallel_shard,
                params.context_parallel_replicate * params.context_parallel_shard,
                params.tensor_parallel
            ),
            mesh_dim_names=(
                'pp',
                'dp',
                'cp',
                'tp'
            )
        )


ALL_DOMAIN_PROVIDERS: list[DeviceMeshDomain] = [RegularDomain(), DenseDomain(), ExpertDomain(), BatchDomain()]
