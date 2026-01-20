import datetime
import logging
import os
import socket
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from torch.distributed import DeviceMesh

from .device_mesh_domains import ALL_DOMAIN_PROVIDERS, REGULAR_DOMAIN
from .log import build_dist_logger

if TYPE_CHECKING:
    from .params import DeviceMeshParameters


def _resolve_master_addr() -> str:
    if "MASTER_ADDR" not in os.environ:
        return "127.0.0.1"

    master_addr = os.environ["MASTER_ADDR"]

    try:
        return socket.gethostbyname(master_addr)
    except OSError:
        return master_addr


def _build_mesh_domains(params: "DeviceMeshParameters") -> dict[str, DeviceMesh]:
    return {
        provider.name: provider.build_mesh(params)
        for provider in ALL_DOMAIN_PROVIDERS
    }


class DistributedContext:
    """
    Acts as the single source of truth for the distributed execution environment.

    It acts as the central repository for the distributed configuration, managing the creation
    and synchronization of PyTorch DeviceMeshes for different domains (Regular domain, Expert Parallel domain, ...).

    All assertions regarding rank placement, group memberships, and parallel topology
    must be derived from this context to ensure consistency.
    """

    def __init__(self, params: "DeviceMeshParameters", log_level: int):
        self._params = params

        if params.is_distributed:
            meshes = _build_mesh_domains(params)
            regular_mesh = meshes[REGULAR_DOMAIN]

            self._meshes = meshes
            self._num_nodes = regular_mesh.size() // torch.cuda.device_count()
            self._logger = build_dist_logger(
                f'pp:{regular_mesh.get_local_rank("pp")}-'
                f'dpr:{regular_mesh.get_local_rank("dp_replicate")}-'
                f'dps:{regular_mesh.get_local_rank("dp_shard")}-'
                f'cps:{regular_mesh.get_local_rank("cp_shard")}-'
                f'cpr:{regular_mesh.get_local_rank("cp_replicate")}-'
                f'tp:{regular_mesh.get_local_rank("tp")}',
                level=log_level
            )
        else:
            self._meshes = None
            self._num_nodes = 1
            self._logger = build_dist_logger("local", level=log_level)

        self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self._global_rank = int(os.environ.get("RANK", "0"))

        self._node_rank = self._global_rank // torch.cuda.device_count()

        self._master_addr = _resolve_master_addr()
        self._current_device = torch.device("cuda")

        torch.cuda.set_device(self._local_rank)

    @property
    def logger(self) -> logging.Logger:
        """Returns the logger instance configured for distributed logging."""

        return self._logger

    def mesh_for(self, domain: str) -> DeviceMesh:
        """
        Returns the device mesh view associated with a specific logical domain.

        Available Domains and Dimensions:
            *   `regular` (`REGULAR_DOMAIN`): The most granular mesh for fully decomposed parallelism.
                Dimensions: ``('pp', 'dp_replicate', 'dp_shard', 'cp_shard', 'cp_replicate', 'tp')``
            *   `expert` (`EXPERT_DOMAIN`): Mesh optimized for distributing MoE (Mixture of Experts) layers.
                Dimensions: ``('pp', 'replicate', 'ep')``
            *   `dense` (`DENSE_DOMAIN`): Mesh optimized for distributing dense layers.
                Dimensions: ``('pp', 'dp_replicate', 'dp_cp_shard', 'cp_replicate', 'tp')``
            *   `batch` (`BATCH_DOMAIN`): Mesh optimized for distributing input data.
                Dimensions: ``('pp', 'dp', 'cp', 'tp')``

        Args:
            domain: The name of the domain to retrieve.

        Returns:
            The PyTorch DeviceMesh configured for the requested domain.

        Raises:
            ValueError: If the specified domain does not exist.
        """

        if domain not in self._meshes:
            raise ValueError(f"Domain {domain} does not exist")
        return self._meshes[domain]

    @property
    def is_main_process(self) -> bool:
        """Checks if the current process is the global rank 0."""

        return self._global_rank == 0

    @property
    def is_local_main_process(self) -> bool:
        """Checks if the current process is the rank 0 on the specific node."""

        return self._local_rank == 0

    def wait_world(self):
        """Blocks process execution until all ranks reach this point."""

        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        torch.cuda.synchronize()

    def set_timeout(self, timeout_seconds: float):
        """
        Updates the NCCL/process group timeout for all underlying meshes.

        Args:
            timeout_seconds: New timeout duration in seconds.
        """

        self.logger.info(f"Setting global timeout to {timeout_seconds} seconds")
        self.wait_world()

        groups: list[torch.distributed.ProcessGroup | None] = [None]
        for mesh in self._meshes.values():
            for dim in range(mesh.ndim):
                groups.append(mesh.get_group(dim))

        for group in groups:
            torch.distributed.distributed_c10d._set_pg_timeout(datetime.timedelta(seconds=timeout_seconds), group)  # noqa: SLF001

    @contextmanager
    def local_main_process_first(self):
        """
        Context manager that executes the block on the local main process first.

        Other local ranks wait at the entrance. The local main process waits at the
        exit to synchronize before continuing.
        """
        if not self.is_local_main_process:
            self.wait_world()

        yield

        if self.is_local_main_process:
            self.wait_world()

    @contextmanager
    def main_process_first(self):
        """
        Context manager that executes the block on the global main process first.

        All other ranks wait at the entrance. The global main process waits at the
        exit to synchronize before continuing.
        """

        if not self.is_main_process:
            self.wait_world()

        yield

        if self.is_main_process:
            self.wait_world()

    @property
    def current_device(self) -> torch.device:
        """Returns the CUDA device associated with this rank."""

        return self._current_device

    @property
    def mesh_params(self) -> "DeviceMeshParameters":
        """Returns the parameters used to initialize this context."""

        return self._params

    @property
    def master_addr(self) -> str:
        """Returns the IP address or domain name of the master node."""

        return self._master_addr

    @property
    def node_rank(self) -> int:
        """Returns the index of the node this process is running on."""

        return self._node_rank

    @property
    def num_nodes(self) -> int:
        """Returns the total number of nodes in the cluster."""

        return self._num_nodes
