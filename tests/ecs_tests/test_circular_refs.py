"""
Test for circular reference handling in Entity system.
"""
import pytest
import logging
from typing import List, Dict, Set, Any, cast
from uuid import UUID

from minference.ecs.entity import Entity
from minference.ecs.enregistry import EntityRegistry
from conftest import ManyToManyLeft, ManyToManyRight

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_circular_refs")

def test_circular_reference_preservation():
    """Test that circular references are preserved during registration and retrieval."""
    # Create entities with circular references
    left = ManyToManyLeft(name="LeftEntity")
    right = ManyToManyRight(name="RightEntity")
    
    # Create circular reference
    left.rights = [right]
    right.lefts = [left]
    
    # Print initial state
    logger.info(f"Initial left: {left}, rights: {left.rights}")
    logger.info(f"Initial right: {right}, lefts: {right.lefts}")
    
    # Initialize dependency graph to handle circular references
    left.initialize_deps_graph()
    
    # Explicitly register with the registry
    # With the new graph-based approach, we need to explicitly register
    EntityRegistry.register(left)
    EntityRegistry.register(right)
    
    # Retrieve from registry
    retrieved_left = ManyToManyLeft.get(left.ecs_id)
    assert retrieved_left is not None
    
    # Verify references are preserved
    logger.info(f"Retrieved left: {retrieved_left}, rights: {retrieved_left.rights}")
    
    # Check that reference to right entity is preserved
    assert len(retrieved_left.rights) == 1
    retrieved_right = retrieved_left.rights[0]
    assert retrieved_right.name == "RightEntity"
    
    # Check that circular reference back to left is preserved
    assert len(retrieved_right.lefts) == 1
    # Check by ID equality rather than object identity, since entities may be deep-copied
    assert retrieved_right.lefts[0].ecs_id == retrieved_left.ecs_id