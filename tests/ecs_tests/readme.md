# Entity Component System Tests

This directory contains tests for the Entity Component System (ECS) with a focus on the core entity functionality, relationship patterns, and dependency graph handling. The tests are designed to verify the correct behavior of the in-memory implementation.

## Test Organization

- **test_basic_operations.py**: Tests for core entity operations like creation, modification detection, forking, etc.
- **test_relationships.py**: Tests for various entity relationship patterns (one-to-one, one-to-many, many-to-many, hierarchical, bidirectional)
- **test_deps_graph.py**: Tests for the dependency graph functionality, including cycle detection and proper handling of complex entity structures
- **conftest.py**: Common fixtures and entity model definitions
- **test_setup.py**: Test setup and utility functions

## Running the Tests

To run all tests:

```bash
python -m pytest tests/ecs_tests/
```

To run specific test files:

```bash
python -m pytest tests/ecs_tests/test_basic_operations.py
python -m pytest tests/ecs_tests/test_relationships.py
python -m pytest tests/ecs_tests/test_deps_graph.py
```

To run with verbose output:

```bash
python -m pytest tests/ecs_tests/ -v
```

## Entity Test Models

The tests use a set of entity models defined in conftest.py:

- **SimpleEntity**: Basic entity with name and value fields
- **ParentEntity**: Entity with one-to-many relationship to SimpleEntity
- **RefEntity**: Entity with one-to-one reference to SimpleEntity
- **ManyToManyLeft/Right**: Entities with many-to-many relationships
- **HierarchicalEntity**: Entity with parent/children references for testing hierarchies
- **BidirectionalParent/Child**: Entities with bidirectional references

Additional models for dependency graph testing are defined in test_deps_graph.py:

- **CircularRefA/B/C**: Entities with circular references
- **DiamondTop/Left/Right/Bottom**: Entities with diamond-shaped dependency pattern

## Test Categories

1. **Entity Creation and Registration**
   - Automatic registration on creation
   - Type-based entity retrieval
   - Cold snapshots vs. warm copies

2. **Modification Detection**
   - Basic field changes
   - Nested entity modifications
   - Collection modifications

3. **Entity Forking**
   - ID and lineage management
   - Data preservation
   - Reference updates

4. **Relationship Patterns**
   - Different relationship types
   - Modification propagation
   - Reference integrity

5. **Dependency Graph**
   - Graph building and structure
   - Cycle detection
   - Topological sorting
   - Complex dependency patterns

6. **Entity Tracer**
   - Automatic tracking and forking
   - Function return value handling

## Notes

- All tests use the in-memory backend for simplicity and focus
- Each test resets the EntityRegistry to ensure test isolation
- Entity test models override the get() method to ensure proper type casting
- Dependency graphs are explicitly initialized to ensure proper relationship tracking