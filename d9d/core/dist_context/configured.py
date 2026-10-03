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
    return {provider.name: provider.build_mesh(params) for provider in ALL_DOMAIN_PROVIDERS}


class DistributedContext:
    """Holds the distributed execution environment and acts as its single source of truth.

    It builds a PyTorch ``DeviceMesh`` for each domain (regular, expert, dense, ...). Rank placement, group
    membership and parallel topology must all come from this context, so that they stay consistent.
    """

    def __init__(self, params: "DeviceMeshParameters", log_level: int):
        """Constructs the ``DistributedContext`` object and sets the current CUDA device to the local rank.

        Args:
            params: The parallelism degrees to build the device meshes from.
            log_level: The log level of the ``d9d`` logger.
        """
        self._params = params

        if params.is_distributed:
            meshes = _build_mesh_domains(params)
            regular_mesh = meshes[REGULAR_DOMAIN]

            self._meshes = meshes
            self._num_nodes = regular_mesh.size() // torch.cuda.device_count()
            self._logger = build_dist_logger(
                f"pp:{regular_mesh.get_local_rank('pp')}-"
                f"dpr:{regular_mesh.get_local_rank('dp_replicate')}-"
                f"dps:{regular_mesh.get_local_rank('dp_shard')}-"
                f"cps:{regular_mesh.get_local_rank('cp_shard')}-"
                f"cpr:{regular_mesh.get_local_rank('cp_replicate')}-"
                f"tp:{regular_mesh.get_local_rank('tp')}",
                level=log_level,
            )
        else:
            self._meshes = {}
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
        """The logger configured for distributed logging."""
        return self._logger

    def mesh_for(self, domain: str) -> DeviceMesh:
        """Returns the device mesh view associated with a specific logical domain.

        The available domains:

        *   ``"regular"`` (``REGULAR_DOMAIN``): the most granular mesh, with every parallelism in its own dimension.
            Dimensions: ``("pp", "dp_replicate", "dp_shard", "cp_shard", "cp_replicate", "tp")``.
        *   ``"expert"`` (``EXPERT_DOMAIN``): the mesh for Mixture-of-Experts (MoE) layers.
            Dimensions: ``("pp", "ep_replicate", "ep_shard")``.
        *   ``"dense"`` (``DENSE_DOMAIN``): the mesh for dense layers.
            Dimensions: ``("pp", "dp_replicate", "dp_cp_shard", "cp_replicate", "tp")``.
        *   ``"batch"`` (``BATCH_DOMAIN``): the mesh for distributing input data.
            Dimensions: ``("pp", "dp", "cp", "tp")``.
        *   ``"flat"`` (``FLAT_DOMAIN``): a single dimension with all the processes.
            Dimensions: ``("world",)``.

        Args:
            domain: The name of the domain.

        Returns:
            The device mesh of the domain.

        Raises:
            ValueError: If the domain does not exist.
        """
        if domain not in self._meshes:
            raise ValueError(
                f"Domain ({domain}) does not exist. Use one of the available domains ({list(self._meshes)})."
            )
        return self._meshes[domain]

    @property
    def is_main_process(self) -> bool:
        """Whether the current process is global rank 0."""
        return self._global_rank == 0

    @property
    def is_local_main_process(self) -> bool:
        """Whether the current process is local rank 0 on its node."""
        return self._local_rank == 0

    def wait_world(self):
        """Blocks until all ranks reach this point and the current CUDA device finishes its queued work."""
        if self._params.is_distributed:
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        torch.cuda.synchronize()

    def set_timeout(self, timeout_seconds: float):
        """Sets the timeout of the default process group and of all mesh process groups.

        Does nothing in a non-distributed setup.

        Args:
            timeout_seconds: The new timeout in seconds.
        """
        if not self._params.is_distributed:
            return

        self.logger.info(f"Setting global timeout to {timeout_seconds} seconds")
        self.wait_world()

        groups: list[torch.distributed.ProcessGroup | None] = [None]
        for mesh in self._meshes.values():
            for dim in range(mesh.ndim):
                groups.append(mesh.get_group(dim))

        for group in groups:
            torch.distributed.set_timeout(datetime.timedelta(seconds=timeout_seconds), group)

    @contextmanager
    def local_main_process_first(self):
        """Runs the block on the local main processes first.

        Other ranks wait before entering the block. Local main processes wait after leaving it, so all ranks
        continue together.
        """
        if not self.is_local_main_process:
            self.wait_world()

        yield

        if self.is_local_main_process:
            self.wait_world()

    @contextmanager
    def main_process_first(self):
        """Runs the block on the global main process first.

        Other ranks wait before entering the block. The global main process waits after leaving it, so all ranks
        continue together.
        """
        if not self.is_main_process:
            self.wait_world()

        yield

        if self.is_main_process:
            self.wait_world()

    @property
    def current_device(self) -> torch.device:
        """The CUDA device of this rank."""
        return self._current_device

    @property
    def mesh_params(self) -> "DeviceMeshParameters":
        """The parameters this context was built from."""
        return self._params

    @property
    def master_addr(self) -> str:
        """The IP address or host name of the master node."""
        return self._master_addr

    @property
    def node_rank(self) -> int:
        """The index of the node this process runs on."""
        return self._node_rank

    @property
    def local_rank(self) -> int:
        """The rank of the current process within its node."""
        return self._local_rank

    @property
    def num_nodes(self) -> int:
        """The total number of nodes in the cluster."""
        return self._num_nodes
