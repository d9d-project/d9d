from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import parallelize_module

from d9d.module.parallelism.style import ToLocalParallel


def parallelize_replicate(
    module: nn.Module,
    mesh: DeviceMesh,
):
    """
    Applies replicated parallelism to the module.

    This function configures the provided module to be fully replicated across the
    given device mesh. It utilizes the ``ToLocalParallel`` style, which manages
    ``DTensor`` wrapping for parameters and gradients (via ``Replicate``
    and ``Partial`` placements) while ensuring that the underlying computation
    sees standard local tensors during the forward pass.

    This approach is effectively Data Parallelism managed via the DTensor
    APIs, allowing seamless integration of modules that require local tensor inputs
    into a broader distributed mesh context.

    Args:
     module: The module to parallelize.
     mesh: The device mesh over which to replicate the module.
    """

    parallelize_module(module, mesh, ToLocalParallel(
        param_placement=tuple(Replicate() for _ in range(mesh.ndim)),
        grad_placement=tuple(Replicate() for _ in range(mesh.ndim))
    ))
