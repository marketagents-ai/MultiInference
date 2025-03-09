# Entity Component System (ECS) Tests

## Overview
This directory contains tests for the Entity Component System (ECS) implementation, focusing on:
1. Basic entity operations
2. Entity relationships with versioning
3. Setup for SQL backend testing

## Key Insights from Testing

### Entity Registration Process
When an entity is registered in `EntityRegistry`, it may automatically fork (create a new version) if modifications are detected. This is why the entity IDs may change during registration.

### Relationship Handling
The key challenge with entity relationships is circular references:

1. **One-to-Many Relationships**: These work well with direct references as long as the parent knows about the children but children don't reference their parents.

2. **Many-to-Many Relationships**: These can cause circular reference issues.
   - **Solution**: Use ID references instead of direct object references:
   ```python
   class ManyToManyRight(Entity):
       name: str
       # Store IDs only
       left_ids: List[UUID] = Field(default_factory=list)
       
       @property
       def lefts(self) -> List["ManyToManyLeft"]:
           """Get the actual objects when needed."""
           return [ManyToManyLeft.get(id) for id in self.left_ids if id]
   ```

3. **Hierarchical Relationships**: These also cause circular reference issues.
   - **Solution**: Similar to many-to-many, use ID references with properties:
   ```python
   class HierarchicalEntity(Entity):
       name: str
       parent_id: Optional[UUID] = None
       child_ids: List[UUID] = Field(default_factory=list)
       
       @property
       def parent(self) -> Optional["HierarchicalEntity"]:
           if self.parent_id:
               return HierarchicalEntity.get(self.parent_id)
           return None
       
       @property
       def children(self) -> List["HierarchicalEntity"]:
           return [HierarchicalEntity.get(id) for id in self.child_ids if id]
   ```

### Entity Registration Flow
Always capture the result of `EntityRegistry.register()` since it might return a different entity than the one provided:

```python
# Wrong: entity might have been forked during registration
EntityRegistry.register(entity)
return entity

# Correct: use the result of registration
result = EntityRegistry.register(entity)
return result
```

### Helper Functions for Testing
For testing, we added:
1. `test_setup.py`: For configuring the registry and storage
2. `debug_entity.py`: For debugging entity creation and storage issues
3. `debug_relationships.py`: For testing different relationship patterns

### Circular Reference Detection
We added cycle detection to the `get_sub_entities()` method to prevent infinite recursion:

```python
def get_subs_with_cycle_detection(entity: Entity, path: Set[UUID]) -> Set[Entity]:
    # Check for circular references
    if entity.ecs_id in path:
        logger.warning(f"Circular reference detected for entity {entity.ecs_id}")
        return set()
        
    # Collect sub-entities with cycle avoidance
    result = set()
    new_path = path.union({entity.ecs_id})
    
    # Process fields with recursion depth limit
    for field_name in entity.model_fields:
        value = getattr(entity, field_name)
        # Handle lists of entities and direct references...
    
    return result
```

## Next Steps for SQLAlchemy Implementation
1. Ensure the SQL models use ID references for relationships to avoid circular dependency issues
2. Store and retrieve UUIDs consistently in the database
3. Implement relationship handling methods that work with both in-memory and SQL storage
4. Optimize for entity trees to load efficiently from storage