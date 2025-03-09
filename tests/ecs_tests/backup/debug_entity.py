"""
Debug script to understand what's happening with entity storage and retrieval.
"""
import sys
import logging
from uuid import UUID, uuid4
from typing import Optional, List, Dict, Any

from minference.ecs.entity import Entity, EntityRegistry, InMemoryEntityStorage
from pydantic import Field

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("EntityDebug")
logger.setLevel(logging.DEBUG)

# Make EntityRegistry available in __main__
sys.modules['__main__'].EntityRegistry = EntityRegistry

# Set up in-memory storage with debug tracing
class DebugInMemoryEntityStorage(InMemoryEntityStorage):
    """Enhanced InMemoryEntityStorage with debug logging."""
    
    def get_cold_snapshot(self, entity_id: UUID) -> Optional[Entity]:
        """Get the cold (stored) version of an entity with detailed logging."""
        logger.debug(f"Getting cold snapshot for entity ID: {entity_id}")
        
        snapshot = self._registry.get(entity_id)
        if snapshot:
            # Print entity details
            logger.debug(f"Found entity: {type(snapshot).__name__}({snapshot.ecs_id})")
            for field_name, field_value in snapshot.__dict__.items():
                if isinstance(field_value, list) and field_value:
                    logger.debug(f"  List field '{field_name}' has {len(field_value)} items")
                    for i, item in enumerate(field_value):
                        logger.debug(f"    Item {i}: {type(item).__name__} ({item.ecs_id if isinstance(item, Entity) else 'not an entity'})")
                elif isinstance(field_value, Entity):
                    logger.debug(f"  Reference field '{field_name}': {type(field_value).__name__}({field_value.ecs_id})")
                elif field_name not in {'_abc_impl', 'model_fields', 'model_config'}:
                    logger.debug(f"  Field '{field_name}': {field_value}")
                    
            # Debug check registry state
            logger.debug(f"Registry has {len(self._registry)} entities")
            logger.debug(f"Registry keys: {list(self._registry.keys())[:5]}{'...' if len(self._registry) > 5 else ''}")
            return snapshot
        else:
            logger.debug(f"No entity found with ID: {entity_id}")
            return None

    def get(self, entity_id: UUID, expected_type: Optional[type] = None) -> Optional[Entity]:
        """Enhanced get method with detailed logging."""
        logger.debug(f"Getting entity ID: {entity_id} (expected type: {expected_type.__name__ if expected_type else 'any'})")
        
        # Get from cold storage
        ent = self._registry.get(entity_id)
        if not ent:
            logger.debug(f"No entity found with ID: {entity_id}")
            return None
            
        logger.debug(f"Found cold entity: {type(ent).__name__}({ent.ecs_id})")
        
        if expected_type and not isinstance(ent, expected_type):
            logger.debug(f"Type mismatch: got {type(ent).__name__}, expected {expected_type.__name__}")
            return None
            
        # Create a warm copy
        logger.debug(f"Creating warm copy of {type(ent).__name__}({ent.ecs_id})")
        warm_copy_before = None
        try:
            import copy
            warm_copy_before = copy.deepcopy(ent)
        except Exception as e:
            logger.error(f"Error in deepcopy (before marking as from_storage): {str(e)}")
        
        # Mark as coming from storage and assign new live_id
        warm_copy = None
        try:
            warm_copy = copy.deepcopy(ent)
            warm_copy.live_id = uuid4()
            warm_copy.from_storage = True
        except Exception as e:
            logger.error(f"Error creating warm copy: {str(e)}")
            return None
        
        # Debug warm copy fields
        logger.debug(f"Warm copy created: {type(warm_copy).__name__}({warm_copy.ecs_id}) with live_id={warm_copy.live_id}")
        for field_name, field_value in warm_copy.__dict__.items():
            if isinstance(field_value, list) and field_value:
                logger.debug(f"  List field '{field_name}' has {len(field_value)} items")
                for i, item in enumerate(field_value):
                    logger.debug(f"    Item {i}: {type(item).__name__} ({item.ecs_id if isinstance(item, Entity) else 'not an entity'})")
            elif isinstance(field_value, Entity):
                logger.debug(f"  Reference field '{field_name}': {type(field_value).__name__}({field_value.ecs_id})")
            elif field_name not in {'_abc_impl', 'model_fields', 'model_config'}:
                logger.debug(f"  Field '{field_name}': {field_value}")
                
        # Debug comparison with cold entity
        if warm_copy_before:
            for field_name, field_value in warm_copy.__dict__.items():
                if field_name not in {'from_storage', 'live_id', '_abc_impl', 'model_fields', 'model_config'}:
                    original_value = getattr(warm_copy_before, field_name, None)
                    if field_value != original_value:
                        logger.debug(f"Field '{field_name}' changed during warm copy creation")
                        logger.debug(f"  Before: {original_value}")
                        logger.debug(f"  After: {field_value}")
        
        return warm_copy

# Define simple test entities
class SimpleEntity(Entity):
    """A simple entity with basic fields for testing."""
    name: str
    value: int = 0

class ParentEntity(Entity):
    """Entity with one-to-many relationship to SimpleEntity."""
    name: str
    children: List[SimpleEntity] = Field(default_factory=list)

# Set up storage
storage = DebugInMemoryEntityStorage()
EntityRegistry.use_storage(storage)

# Create test entities
logger.info("Creating test entities")
child1 = SimpleEntity(name="Child1", value=1)
child2 = SimpleEntity(name="Child2", value=2)
logger.info(f"Created child1: {child1.ecs_id}")
logger.info(f"Created child2: {child2.ecs_id}")

# Explicitly register children
EntityRegistry.register(child1)
EntityRegistry.register(child2)
logger.info("Registered children")

parent = ParentEntity(name="Parent")
parent.children = [child1, child2]
logger.info(f"Created parent: {parent.ecs_id}")

# Register parent with its children
EntityRegistry.register(parent)
logger.info("Registered parent")

# Check the registry contents
logger.info("Registry state:")
for entity_id, entity in storage._registry.items():
    logger.info(f"  {entity_id}: {type(entity).__name__}, children={len(getattr(entity, 'children', []))}")

# Attempt to retrieve the parent entity
logger.info("Retrieving parent entity")
retrieved_parent = ParentEntity.get(parent.ecs_id)

if retrieved_parent:
    logger.info(f"Retrieved parent: {retrieved_parent.ecs_id}, children count: {len(retrieved_parent.children)}")
    for i, child in enumerate(retrieved_parent.children):
        logger.info(f"  Child {i}: {child.name}, value={child.value}")
else:
    logger.error("Failed to retrieve parent entity")

# Check cold snapshot directly
logger.info("Retrieving cold snapshot of parent")
cold_parent = EntityRegistry.get_cold_snapshot(parent.ecs_id)
if cold_parent:
    logger.info(f"Cold parent: {cold_parent.ecs_id}, children count: {len(getattr(cold_parent, 'children', []))}")
    for i, child in enumerate(getattr(cold_parent, 'children', [])):
        logger.info(f"  Child {i}: {child.name}, value={child.value}")
else:
    logger.error("Failed to retrieve cold snapshot of parent")

logger.info("Debug complete")