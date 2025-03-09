"""
Entity dependency graph implementation.

This module provides utilities to compute the dependency graph of entities
and detect circular references without modifying the underlying object model.
"""
from .graph import EntityDependencyGraph, CycleStatus, GraphNode

__all__ = ["EntityDependencyGraph", "CycleStatus", "GraphNode"]