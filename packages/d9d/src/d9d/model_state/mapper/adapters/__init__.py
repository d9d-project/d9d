"""Adapters that build simple ``ModelStateMapper`` instances from PyTorch modules or other mappers."""

from .mapper import identity_mapper_from_mapper_outputs
from .module import identity_mapper_from_module

__all__ = [
    "identity_mapper_from_mapper_outputs",
    "identity_mapper_from_module",
]
