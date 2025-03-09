"""
Setup and utility functions for ECS tests.
"""
import sys
import logging
from uuid import uuid4

from minference.ecs.entity import Entity, EntityRegistry, InMemoryEntityStorage

# Configure logging for tests
def setup_ecs_tests():
    """Set up the test environment for ECS tests."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Make EntityRegistry available in __main__
    sys.modules['__main__'].EntityRegistry = EntityRegistry
    
    # Use in-memory storage
    storage = InMemoryEntityStorage()
    EntityRegistry.use_storage(storage)
    
    return storage

def cleanup_ecs_tests():
    """Clean up after ECS tests."""
    # Clear the registry
    EntityRegistry.clear()
    
    # Remove from __main__ if needed
    if hasattr(sys.modules['__main__'], 'EntityRegistry'):
        delattr(sys.modules['__main__'], 'EntityRegistry')