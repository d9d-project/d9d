"""Grouped GEMM autograd functions with explicit gradient control, one subpackage per kernel library.

Subpackages are not re-exported here: ``nv`` and ``quack`` need optional dependencies, so they are imported directly.
"""
