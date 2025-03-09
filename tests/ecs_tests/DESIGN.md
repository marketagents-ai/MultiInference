# ECS System Test Design

## Overview

This testing strategy systematically verifies the Entity Component System (ECS) functionality, focusing first on the in-memory implementation and then extending to SQLAlchemy-based persistence. The goal is to establish a robust test suite that validates entity versioning, relationship management, and hierarchical change propagation across different storage backends.

## Testing Strategy

### Phase 1: In-Memory Implementation

We'll first create tests for the in-memory implementation, which is already functioning correctly. These tests will serve as the reference point for validating the SQL implementation later.

### Phase 2: SQLAlchemy Implementation

After establishing the in-memory tests, we'll add a parallel set of tests for the SQLAlchemy backend, ensuring both implementations behave identically.

### Phase 3: Production Models

Finally, we'll apply the learned patterns to the actual production models in models.py and ensure they work correctly with both backends.

## Test Categories

### 1. Basic Entity Operations

- **Entity Creation and Registration**: Testing entity creation and automatic registration
- **Entity Retrieval**: Getting entities by ID, type, and other criteria
- **Entity Modification**: Changing entity fields and detecting modifications
- **Entity Forking**: Creating new versions when changes are detected
- **Lineage Tracking**: Verifying parent-child relationships between versions

### 2. Relationship Patterns

#### 2.1 One-to-One Relationships

- Simple direct reference (A -> B)
- Bidirectional references (A <-> B)
- Optional references (A -> ?B)

#### 2.2 One-to-Many Relationships

- Parent with children list (Parent -> [Child1, Child2, ...])
- Bidirectional parent-child (Parent <-> Children)
- Sorted collections (ordered relationships)

#### 2.3 Many-to-Many Relationships

- Simple many-to-many (A <-> B)
- Many-to-many with association data
- Self-referential many-to-many (A <-> A)

#### 2.4 Hierarchical Relationships

- Multi-level hierarchies (e.g., A -> B -> C -> D)
- Circular references (A -> B -> C -> A)
- Polymorphic relationships (relationship with different entity types)

### 3. Entity Tracer Testing

- Functions decorated with `@entity_tracer`
- Automatic detection of entity changes
- Proper forking of changed entities
- Bottom-up propagation of changes

### 4. Complex Scenarios

- Concurrent modifications to related entities
- Deep nested structures with mixed relationship types
- Large entity graphs with many interconnections
- Performance testing with many entities

## Test Implementation Approach

### Mock Entity Classes

We'll create specialized mock entity classes for testing:

```python
class SimpleEntity(Entity):
    name: str
    value: int

class ParentEntity(Entity):
    name: str
    children: List[SimpleEntity] = Field(default_factory=list)

class ManyToManyLeft(Entity):
    name: str
    rights: List["ManyToManyRight"] = Field(default_factory=list)

class ManyToManyRight(Entity):
    name: str
    lefts: List["ManyToManyLeft"] = Field(default_factory=list)
```

### Fixture Organization

Tests will use fixtures to set up entity configurations:

```python
@pytest.fixture
def simple_entity():
    return SimpleEntity(name="Test", value=42)
    
@pytest.fixture
def parent_with_children():
    parent = ParentEntity(name="Parent")
    child1 = SimpleEntity(name="Child1", value=1)
    child2 = SimpleEntity(name="Child2", value=2)
    parent.children = [child1, child2]
    return parent
```

### Test Structure

Tests will be organized by relationship pattern and complexity level:

```
tests/ecs_tests/
  ├── test_basic_operations.py     # Basic entity operations
  ├── test_one_to_one.py           # One-to-one relationships
  ├── test_one_to_many.py          # One-to-many relationships
  ├── test_many_to_many.py         # Many-to-many relationships
  ├── test_hierarchical.py         # Hierarchical relationships
  ├── test_entity_tracer.py        # Entity tracer functionality
  ├── test_complex_scenarios.py    # Complex mixed scenarios
  └── conftest.py                  # Shared fixtures
```

### Backend Switching

Tests will support both in-memory and SQL backends:

```python
@pytest.fixture
def storage_backend(request):
    """Set up storage backend based on parameter."""
    backend_type = request.param
    
    if backend_type == "memory":
        # Set up in-memory backend
        storage = InMemoryEntityStorage()
        EntityRegistry.use_storage(storage)
    elif backend_type == "sql":
        # Set up SQL backend
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine)
        storage = SqlEntityStorage(session_factory, MOCK_ENTITY_ORM_MAP)
        EntityRegistry.use_storage(storage)
    
    yield backend_type
    
    # Clean up after test
    EntityRegistry.clear()

@pytest.mark.parametrize("storage_backend", ["memory", "sql"], indirect=True)
def test_entity_creation(storage_backend):
    """Test entity creation works the same on both backends."""
    # Test code here will run twice, once for each backend
```

## Sample Test Cases

### Basic Entity Lifecycle

```python
def test_entity_lifecycle(storage_backend):
    """Test basic entity lifecycle: create, modify, fork."""
    # Create entity
    entity = SimpleEntity(name="Test", value=42)
    
    # Verify registration
    retrieved = SimpleEntity.get(entity.ecs_id)
    assert retrieved.name == "Test"
    assert retrieved.value == 42
    
    # Modify and check changes
    retrieved.value = 100
    has_changes, _ = retrieved.has_modifications(entity)
    assert has_changes
    
    # Fork to create new version
    new_version = retrieved.fork()
    assert new_version.ecs_id != entity.ecs_id
    assert new_version.lineage_id == entity.lineage_id
    assert new_version.parent_id == entity.ecs_id
    assert new_version.value == 100
```

### One-to-Many Relationship with Change Propagation

```python
def test_child_change_propagation(storage_backend):
    """Test that changes to children propagate to parent."""
    # Create parent with children
    parent = ParentEntity(name="Parent")
    child1 = SimpleEntity(name="Child1", value=1)
    child2 = SimpleEntity(name="Child2", value=2)
    parent.children = [child1, child2]
    
    # Retrieve and modify child
    retrieved_parent = ParentEntity.get(parent.ecs_id)
    retrieved_child = retrieved_parent.children[0]
    retrieved_child.value = 100
    
    # Verify parent detects changes
    has_changes, modified = retrieved_parent.has_modifications(parent)
    assert has_changes
    assert retrieved_child in modified
    
    # Fork and verify new versions
    new_parent = retrieved_parent.fork()
    assert new_parent.ecs_id != parent.ecs_id
    assert new_parent.children[0].value == 100
    assert new_parent.children[0].ecs_id != child1.ecs_id
```

## Implementation Timeline

1. **Week 1**: Set up test infrastructure and implement basic entity operation tests
2. **Week 2**: Implement relationship pattern tests for the in-memory backend
3. **Week 3**: Develop the SQLAlchemy backend implementation
4. **Week 4**: Extend tests to cover SQLAlchemy backend
5. **Week 5**: Apply learned patterns to refactor production models

## Expected Outcomes

- A comprehensive test suite that validates ECS functionality
- Clear documentation of expected behavior for all relationship patterns
- A robust SQLAlchemy implementation that matches in-memory behavior
- Guidelines for implementing entity-to-ORM mappings
- Performance metrics for both storage backends