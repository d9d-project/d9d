import datetime
import logging
import os
import socket
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from torch.distributed import init_device_mesh, DeviceMesh

from d9d.core.dist_context.log import build_dist_logger

if TYPE_CHECKING:
    from d9d.core.dist_context.params import DeviceMeshParameters


def _resolve_master_addr() -> str:
    if 'MASTER_ADDR' not in os.environ:
        return '127.0.0.1'

    master_addr = os.environ['MASTER_ADDR']

    try:
        return socket.gethostbyname(master_addr)
    except OSError:
        return master_addr


# TODO idea: create DeviceMesh domain abstraction i.e. experts domain, general domain, ...
# TODO idea: would be useful for training non-sequential models in future
# TODO: we currently do not support TP


class DistributedContext:
    """
    Acts as the single source of truth for the distributed execution environment.

    It acts as the central repository for the distributed configuration, managing the creation
    and synchronization of PyTorch DeviceMeshes for different domains (Regular domain, Expert Parallel domain, ...).

    All assertions regarding rank placement, group memberships, and parallel topology
    must be derived from this context to ensure consistency.
    """

    def __init__(self, params: 'DeviceMeshParameters'):
        self._params = params

        if params.is_distributed:
            ep_replicate_dim = (params.data_parallel_replicate * params.data_parallel_shard * params.context_parallel
                                // params.expert_parallel)
            self._mesh_ep = init_device_mesh(
                device_type="cuda",
                mesh_shape=(params.pipeline_parallel,
                            ep_replicate_dim,
                            params.expert_parallel),
                mesh_dim_names=('pp', 'replicate', 'ep')
            )
            self._mesh_regular = init_device_mesh(
                device_type="cuda",
                mesh_shape=(params.pipeline_parallel, params.data_parallel_replicate, params.data_parallel_shard,
                            params.context_parallel),
                mesh_dim_names=('pp', 'dp_replicate', 'dp_shard', 'cp')
            )
            self._num_nodes = self._mesh_regular.size() // torch.cuda.device_count()
            self._logger = build_dist_logger(
                f'pp:{self._mesh_regular.get_local_rank("pp")}-'
                f'dpr:{self._mesh_regular.get_local_rank("dp_replicate")}-'
                f'dps:{self._mesh_regular.get_local_rank("dp_shard")}-'
                f'cp:{self._mesh_regular.get_local_rank("cp")}-'
            )
        else:
            self._mesh_ep = None
            self._mesh_regular = None
            self._num_nodes = 1
            self._logger = build_dist_logger('local')


        self._local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self._global_rank = int(os.environ.get('RANK', '0'))

        self._node_rank = self._global_rank // torch.cuda.device_count()

        self._master_addr = _resolve_master_addr()
        self._current_device = torch.device("cuda")

        torch.cuda.set_device(self._local_rank)

    @property
    def logger(self) -> logging.Logger:
        """Returns the logger instance configured for distributed logging."""

        return self._logger

    @property
    def mesh_ep(self) -> DeviceMesh | None:
        """Returns the device mesh related to expert layers domain for Expert Parallel."""

        return self._mesh_ep

    @property
    def mesh_regular(self) -> DeviceMesh | None:
        """Returns the device mesh related to regular model domain."""

        return self._mesh_regular

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

        self.logger.info(f'Setting global timeout to {timeout_seconds} seconds')
        self.wait_world()

        groups: list[torch.distributed.ProcessGroup | None] = [None]
        for mesh in self.mesh_regular, self.mesh_ep:
            for dim in range(mesh.ndim):
                groups.append(mesh.get_group(dim))

        for group in groups:
            torch.distributed.distributed_c10d._set_pg_timeout(datetime.timedelta(seconds=timeout_seconds), group)

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
    def mesh_params(self) -> DeviceMeshParameters:
        """Returns the parameters used to initialize this context."""

        return self._params

    @property
    def dp_size(self) -> int:
        """Returns the total number of Data Parallel ranks (Replicated * Sharded)."""

        return self.mesh_regular['dp_replicate'].size() * self.mesh_regular['dp_shard'].size()

    @property
    def dp_rank(self) -> int:
        """Returns the linearized Data Parallel rank."""

        return (self.mesh_regular['dp_shard'].size() * self.mesh_regular['dp_replicate'].get_local_rank() +
                self.mesh_regular['dp_shard'].get_local_rank())

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
