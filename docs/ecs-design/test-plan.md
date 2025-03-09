# ECS Testing Plan

## Goals

1. Create a systematic test suite for the Entity Component System
2. Focus on the in-memory implementation only (no SQL)
3. Test core features and all relationship patterns
4. Ensure proper dependency graph handling
5. Make tests clear, maintainable, and independent

## Test Structure

### Test Files Organization

```
tests/
├── unit/
│   ├── ecs/
│   │   ├── conftest.py              # Common fixtures and entity definitions
│   │   ├── test_entity_basics.py    # Core entity functionality
│   │   ├── test_entity_registry.py  # Registry operations
│   │   ├── test_entity_tracer.py    # Entity tracer decorator
│   │   ├── test_deps_graph.py       # Dependency graph features
│   │   └── test_relationships/      # Relationship patterns
│   │       ├── test_one_to_one.py
│   │       ├── test_one_to_many.py
│   │       ├── test_many_to_many.py
│   │       └── test_hierarchical.py
```

### Test Categories

1. **Entity Core Features**
   - Initialization and attributes
   - ID and lineage tracking
   - Serialization and deserialization
   - Cold/warm copy behavior

2. **Entity Registry**
   - Registration
   - Retrieval (by ID, type, etc.)
   - Status reporting

3. **Modification Detection**
   - Field changes
   - Reference changes
   - Collection changes

4. **Entity Forking**
   - Basic forking
   - Parent-child relationship preservation
   - ID and lineage management

5. **Relationship Patterns**
   - One-to-one
   - One-to-many
   - Many-to-many
   - Hierarchical

6. **Dependency Graph**
   - Graph building
   - Cycle detection
   - Topological sorting
   - Dependency queries

7. **Entity Tracer**
   - Automatic tracking
   - Forking behavior
   - Function return value handling

## Test Implementation Details

### Test Entity Classes

We'll create simple, focused entity classes for testing:

```python
class SimpleEntity(Entity):
    """Basic entity with simple attributes"""
    name: str
    value: int = 0

class OneToOneParent(Entity):
    """Entity with a single reference to another entity"""
    name: str
    child: Optional[SimpleEntity] = None

class OneToManyParent(Entity):
    """Entity with a list of child entities"""
    name: str
    children: List[SimpleEntity] = Field(default_factory=list)

class ManyToManyEntity(Entity):
    """Entity with a many-to-many relationship"""
    name: str
    related: List["ManyToManyEntity"] = Field(default_factory=list)

class HierarchicalEntity(Entity):
    """Entity with parent and children for deep hierarchies"""
    name: str
    parent: Optional["HierarchicalEntity"] = None
    children: List["HierarchicalEntity"] = Field(default_factory=list)
```

### Common Test Patterns

For each test, we'll follow this general pattern:

1. **Arrange**: Set up entities with the necessary structure
2. **Act**: Perform operations on entities
3. **Assert**: Verify expected behavior
4. **Clean up**: Reset the registry after each test

### Type Safety

All tests will maintain strong typing:

- Use proper type variables
- Override get() method to return the correct types
- Use cast() where appropriate
- Include null checks before attribute access

### Fixtures

We'll utilize pytest fixtures for:

- Entity creation
- Common entity structures
- Registry setup and teardown
- Test isolation

## Test Cases Overview

### Basic Entity Operations

1. **Creation and Registration**
   - Verify automatic registration on creation
   - Test ID and lineage ID generation

2. **Entity Retrieval**
   - Get by ID
   - Get by type
   - Get multiple by IDs

3. **Cold and Warm Copies**
   - Test difference between retrieved and stored entities
   - Verify live_id vs. ecs_id behavior

### Modification Detection

1. **Field Modifications**
   - Change primitive fields
   - Verify detection through has_modifications()
   - Check returned modified entities

2. **Nested Modifications**
   - Change fields in referenced entities
   - Verify detection in parent entities

3. **Collection Modifications**
   - Add/remove items from lists
   - Rearrange items
   - Modify items in collections

### Entity Forking

1. **Basic Forking**
   - Test forking creates new ID
   - Verify parent ID and lineage IDs
   - Check old_ids list

2. **Nested Forking**
   - Test forking with nested entities
   - Verify all affected entities get new IDs
   - Test reference updates

### Relationship Tests

For each relationship pattern, test:

1. **Creation and Storage**
   - Create related entities
   - Verify registration
   - Verify retrieval with relationships intact

2. **Modification**
   - Modify related entities
   - Verify change detection
   - Test forking with related entities

3. **Reassignment**
   - Replace references
   - Add/remove from collections
   - Test impact on change detection

### Dependency Graph Tests

1. **Graph Building**
   - Test graph initialization
   - Verify node creation
   - Check dependency relationships

2. **Cycle Detection**
   - Create circular references
   - Test cycle detection
   - Verify handling of cycles

3. **Topological Sorting**
   - Create dependency trees
   - Test sorted order
   - Verify dependencies come before dependents

### Entity Tracer Tests

1. **Basic Tracing**
   - Test decorator with simple function
   - Verify automatic forking
   - Check returned entity

2. **Multiple Entity Arguments**
   - Test with multiple entity arguments
   - Verify all are properly tracked
   - Test with mix of entity and non-entity arguments

3. **Async Functions**
   - Test with async functions
   - Verify proper wrapping and execution

## Implementation Approach

1. Start with basic entity tests
2. Implement relationship tests
3. Add dependency graph tests
4. Finish with entity tracer tests

Each test should be focused, descriptive, and independent.