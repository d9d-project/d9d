import datetime
import logging
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from types import TracebackType
from typing import Self

import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh

from .domain import DeviceMeshDomain, DimT, MeshDim

logger = logging.getLogger(__name__)


class DistributedContext:
    """The distributed state of one process: its rank, its GPU, its process groups and the meshes of its domains.

    Create it with ``init_dist_context()``. The code that creates it closes it.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        local_rank: int,
        device: torch.device,
        domain_dims: Mapping[DeviceMeshDomain, tuple[MeshDim, ...]],
        meshes: Mapping[DeviceMeshDomain, DeviceMesh],
        groups: list[dist.ProcessGroup],
        owns_default_group: bool,
    ):
        """Constructs the ``DistributedContext`` object.

        Call ``init_dist_context()`` instead, which builds the arguments.

        Args:
            rank: The global rank of this process.
            world_size: The number of ranks in the job.
            local_rank: The index of this rank on its node.
            device: The GPU of this rank.
            domain_dims: The dimensions of the mesh of every domain of the context.
            meshes: The mesh of every domain. Empty in a job without parallelism.
            groups: The process groups that the context created and destroys on close.
            owns_default_group: Whether the context created the default process group and destroys it on close.
        """
        self._rank = rank
        self._world_size = world_size
        self._local_rank = local_rank
        self._device = device
        self._domain_dims = dict(domain_dims)
        self._meshes = dict(meshes)
        self._groups = list(groups)
        self._owns_default_group = owns_default_group
        self._closed = False

    def _check_open(self):
        if self._closed:
            raise RuntimeError("The distributed context is closed. Use it only before close().")

    def _dims_of(self, domain: DeviceMeshDomain) -> tuple[MeshDim, ...]:
        if domain in self._domain_dims:
            return self._domain_dims[domain]
        raise ValueError(
            f"The context has no domain ({domain.name}). For a built-in domain, set its section of the mesh "
            f"configuration, e.g. mesh.expert. Otherwise pass this domain object in extra_domains of "
            f"init_dist_context()."
        )

    def mesh_for(self, domain: DeviceMeshDomain) -> DeviceMesh:
        """Returns the device mesh of a domain.

        Args:
            domain: A domain of this context.

        Returns:
            The device mesh of the domain.

        Raises:
            ValueError: If the job has no parallelism or the context has no such domain.
            RuntimeError: If the context is closed.
        """
        self._check_open()
        self._dims_of(domain)
        if not self.is_distributed:
            raise ValueError(
                "The job has no parallelism, so it has no device meshes. Check is_distributed before mesh_for()."
            )
        return self._meshes[domain]

    def coordinate(self, domain: DeviceMeshDomain[DimT], dim: DimT) -> int:
        """Returns the coordinate of this rank along a dimension of the mesh of a domain.

        Args:
            domain: A domain of this context.
            dim: The name of a dimension of the domain.

        Returns:
            The coordinate, 0 in a job without parallelism.

        Raises:
            ValueError: If the context has no such domain or the domain has no such dimension.
            RuntimeError: If the context is closed.
        """
        self._dim_index(domain, self._dims_of(domain), dim)
        if not self.is_distributed:
            return 0
        return self.mesh_for(domain).get_local_rank(dim)

    def dim_size(self, domain: DeviceMeshDomain[DimT], dim: DimT) -> int:
        """Returns the number of ranks along a dimension of the mesh of a domain, 1 in a job without parallelism.

        Args:
            domain: A domain of this context.
            dim: The name of a dimension of the domain.

        Returns:
            The number of ranks along the dimension.

        Raises:
            ValueError: If the context has no such domain or the domain has no such dimension.
        """
        dims = self._dims_of(domain)
        return dims[self._dim_index(domain, dims, dim)].size

    @staticmethod
    def _dim_index(domain: DeviceMeshDomain, dims: tuple[MeshDim, ...], dim: str) -> int:
        for index, item in enumerate(dims):
            if item.name == dim:
                return index
        raise ValueError(
            f"The domain ({domain.name}) has no dimension ({dim}). Use one of {[str(item.name) for item in dims]}."
        )

    @property
    def is_distributed(self) -> bool:
        """Whether the job has more than one rank."""
        return self._world_size > 1

    @property
    def rank(self) -> int:
        """The global rank of this process."""
        return self._rank

    @property
    def world_size(self) -> int:
        """The number of ranks in the job."""
        return self._world_size

    @property
    def is_main_process(self) -> bool:
        """Whether the current process is global rank 0."""
        return self._rank == 0

    @property
    def local_rank(self) -> int:
        """The index of this rank on its node."""
        return self._local_rank

    @property
    def is_local_main_process(self) -> bool:
        """Whether the current process is local rank 0 on its node."""
        return self._local_rank == 0

    @property
    def current_device(self) -> torch.device:
        """The GPU of this rank, with its index."""
        return self._device

    def wait_world(self):
        """Blocks until all ranks reach this point and the GPU of this rank finishes its queued work.

        Raises:
            RuntimeError: If the context is closed.
        """
        self._check_open()
        if self.is_distributed:
            dist.barrier(device_ids=[self._device.index])
        torch.cuda.synchronize(self._device)

    def set_timeout(self, timeout_seconds: float):
        """Sets the timeout of the default process group and of all process groups of the context.

        Does nothing in a job without parallelism.

        Args:
            timeout_seconds: The new timeout in seconds.

        Raises:
            RuntimeError: If the context is closed.
        """
        self._check_open()
        if not self.is_distributed:
            return

        logger.info(f"Setting global timeout to {timeout_seconds} seconds")
        self.wait_world()

        timeout = datetime.timedelta(seconds=timeout_seconds)
        dist.set_timeout(timeout)
        for group in self._groups:
            dist.set_timeout(timeout, group)

    @contextmanager
    def local_main_process_first(self) -> Generator[None, None, None]:
        """Runs the block on the local main processes first.

        Other ranks wait before entering the block. Local main processes wait after leaving it, so all ranks
        continue together.

        Raises:
            RuntimeError: If the context is closed.
        """
        self._check_open()
        if not self.is_local_main_process:
            self.wait_world()

        yield

        if self.is_local_main_process:
            self.wait_world()

    @contextmanager
    def main_process_first(self) -> Generator[None, None, None]:
        """Runs the block on the global main process first.

        Other ranks wait before entering the block. The global main process waits after leaving it, so all ranks
        continue together.

        Raises:
            RuntimeError: If the context is closed.
        """
        self._check_open()
        if not self.is_main_process:
            self.wait_world()

        yield

        if self.is_main_process:
            self.wait_world()

    def close(self):
        """Destroys the process groups that the context created.

        It destroys the default process group only if the context created it. A second call does nothing.
        """
        if self._closed:
            return
        self._closed = True
        self._meshes.clear()
        groups, self._groups = self._groups, []
        # Destroying the default group destroys all groups, e.g. if the script already called
        # destroy_process_group().
        if not dist.is_initialized():
            return
        # Reverse order, as destroy_process_group() does: ncclCommAbort() is collective in some NCCL versions.
        for group in reversed(groups):
            dist.destroy_process_group(group)
        if self._owns_default_group:
            dist.destroy_process_group()

    def __enter__(self) -> Self:
        """Returns this context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ):
        """Closes this context.

        If the block raised, the context keeps its process groups and only stops working, as after ``close()``.
        """
        if exc_value is None:
            self.close()
            return
        # Destroying a group waits for the other ranks, which can wait in a collective with this rank, so the job
        # would hang until the timeout. The process exits with the error instead, and the launcher stops the other
        # ranks. torchtitan also destroys its groups only after a successful run (torchtitan/train.py).
        self._closed = True
        self._meshes.clear()
        self._groups = []
