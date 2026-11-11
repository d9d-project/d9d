from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate

from .placement import distribute_parameters


def parallelize_replicate(module: nn.Module, mesh: DeviceMesh):
    """Replicates the parameters of a module for data parallelism.

    Each parameter becomes a ``DTensor`` with ``Replicate`` placements on every mesh dimension. The module still
    computes with plain local tensors, and the ``ParameterRuntime`` all-reduces the gradients.

    Args:
        module: The module to parallelize.
        mesh: The device mesh to replicate the module over.
    """
    distribute_parameters(module, mesh, tuple(Replicate() for _ in range(mesh.ndim)))
