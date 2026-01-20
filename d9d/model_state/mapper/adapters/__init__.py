"""
This package provides utility functions that are used to create simple ModelStateMapper instances from objects
such as PyTorch modules or other StateMappers
"""

from .mapper import identity_mapper_from_mapper_outputs
from .module import identity_mapper_from_module

__all__ = [
    "identity_mapper_from_mapper_outputs",
    "identity_mapper_from_module"
]
