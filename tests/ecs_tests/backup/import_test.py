"""
Test file to verify entity imports without modifying source code.
"""
import sys
from minference.ecs.entity import EntityRegistry, InMemoryEntityStorage, Entity
from minference.ecs.entity import compare_entity_fields, _collect_entities, _check_and_process_entities
from uuid import UUID, uuid4
from pydantic import Field

# Make EntityRegistry available in __main__ for Entity methods to find
sys.modules['__main__'].EntityRegistry = EntityRegistry

# Set up in-memory storage
storage = InMemoryEntityStorage()
EntityRegistry.use_storage(storage)

class SimpleEntity(Entity):
    """A simple entity with basic fields for testing."""
    name: str
    value: int = 0

# Try creating a simple entity
entity = SimpleEntity(name="Test", value=42)
print(f"Created entity: {entity.ecs_id}")

# Retrieve entity from registry
retrieved = EntityRegistry.get(entity.ecs_id)
print(f"Retrieved entity: {retrieved and retrieved.ecs_id}")

if retrieved:
    print(f"Entity name: {retrieved.name}, value: {retrieved.value}")
    
# Clean up
EntityRegistry.clear()
print("Test completed successfully")