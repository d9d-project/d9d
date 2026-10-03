from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import parallelize_module

from d9d.module.parallelism.style import ToLocalParallel


def parallelize_replicate(
    module: nn.Module,
    mesh: DeviceMesh,
):
    """Applies Replicate Parallelism to a module.

    Parameters become ``DTensor`` objects with ``Replicate`` placements on every mesh dimension.
    During the forward pass, the module computes with plain local tensors (see ``ToLocalParallel``).
    This is data parallelism expressed with ``DTensor``.

    Args:
        module: The module to parallelize.
        mesh: The device mesh to replicate the module over.
    """
    parallelize_module(
        module,
        mesh,
        ToLocalParallel(
            param_placement=tuple(Replicate() for _ in range(mesh.ndim)),
            grad_placement=tuple(Replicate() for _ in range(mesh.ndim)),
        ),
    )
