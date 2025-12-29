"""
This package provides utility functions that are used to create simple ModelStateMapper instances from objects
such as PyTorch modules or other StateMappers
"""

from .module import identity_mapper_from_module
from .mapper import identity_mapper_from_mapper_outputs

__all__ = [
    "identity_mapper_from_module",
    "identity_mapper_from_mapper_outputs"
]
