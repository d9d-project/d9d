"""
This package configures the distributed environment and device meshes.
"""

from .configured import DistributedContext
from .device_mesh_domains import BATCH_DOMAIN, DENSE_DOMAIN, EXPERT_DOMAIN, FLAT_DOMAIN, REGULAR_DOMAIN
from .log import build_dist_logger
from .params import DeviceMeshParameters

__all__ = [
    "BATCH_DOMAIN",
    "DENSE_DOMAIN",
    "EXPERT_DOMAIN",
    "FLAT_DOMAIN",
    "REGULAR_DOMAIN",
    "DeviceMeshParameters",
    "DistributedContext",
    "build_dist_logger"
]
