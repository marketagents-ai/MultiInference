"""
Tests for basic entity operations in the ECS system.
"""
import pytest
from typing import cast, List, Dict, Set
from uuid import UUID, uuid4

from minference.ecs.entity import Entity
from minference.ecs.enregistry import EntityRegistry, entity_tracer
from minference.ecs.dependency.graph import EntityDependencyGraph

from conftest import SimpleEntity, ParentEntity

class TestEntityCreation:
    """Tests for entity creation and registration."""
    
    def test_entity_creation_and_registration(self):
        """Test that entities are automatically registered on creation."""
        entity = SimpleEntity(name="Test", value=42)
        entity.initialize_deps_graph()
        
        # Entity should be retrievable from registry
        retrieved = EntityRegistry.get(entity.ecs_id)
        assert retrieved is not None
        assert retrieved.ecs_id == entity.ecs_id
        assert retrieved.name == "Test"
        assert retrieved.value == 42
    
    def test_entity_type_filtering(self):
        """Test retrieving entities by type."""
        # Create different entity types
        simple1 = SimpleEntity(name="Simple1", value=1)
        simple1.initialize_deps_graph()
        
        simple2 = SimpleEntity(name="Simple2", value=2)
        simple2.initialize_deps_graph()
        
        parent = ParentEntity(name="Parent")
        parent.initialize_deps_graph()
        
        # Retrieve by type
        simple_entities = EntityRegistry.list_by_type(SimpleEntity)
        parent_entities = EntityRegistry.list_by_type(ParentEntity)
        
        # Verify results
        assert len(simple_entities) == 2
        assert len(parent_entities) == 1
        assert all(isinstance(e, SimpleEntity) for e in simple_entities)
        assert all(isinstance(e, ParentEntity) for e in parent_entities)
    
    def test_cold_warm_copies(self):
        """Test differences between cold snapshots and warm copies."""
        entity = SimpleEntity(name="Original", value=10)
        entity.initialize_deps_graph()
        
        # Get a warm copy
        warm_copy = SimpleEntity.get(entity.ecs_id)
        assert warm_copy is not None
        
        # Verify warm copy has different live_id but same ecs_id
        assert warm_copy.ecs_id == entity.ecs_id
        assert warm_copy.live_id != entity.live_id
        
        # Get cold snapshot directly
        cold_snapshot = EntityRegistry.get_cold_snapshot(entity.ecs_id)
        assert cold_snapshot is not None
        cold_entity = cast(SimpleEntity, cold_snapshot)
        
        # Verify warm copy vs cold snapshot
        assert cold_entity.ecs_id == warm_copy.ecs_id
        assert cold_entity.from_storage == False
        assert warm_copy.from_storage == True

class TestModificationDetection:
    """Tests for entity modification detection."""
    
    def test_entity_modification_detection(self):
        """Test that entity modifications are properly detected."""
        entity = SimpleEntity(name="Original", value=10)
        entity.initialize_deps_graph()
        
        # Get a copy and modify it
        retrieved = SimpleEntity.get(entity.ecs_id)
        assert retrieved is not None
        retrieved.value = 20
        
        # Check modifications
        has_changes, modified_entities = retrieved.has_modifications(entity)
        assert has_changes
        assert retrieved in modified_entities

class TestEntityForking:
    """Tests for entity forking functionality."""
    
    def test_entity_forking(self):
        """Test that forking creates a new entity version with correct relationships."""
        # Create the initial entity
        entity = SimpleEntity(name="Original", value=10)
        entity.initialize_deps_graph()
        
        # Get the stored version
        stored_entity = EntityRegistry.get_cold_snapshot(entity.ecs_id)
        assert stored_entity is not None
        
        # Create a different copy with modifications
        modified_entity = SimpleEntity.get(entity.ecs_id)
        assert modified_entity is not None
        modified_entity.value = 20
        
        # Fork the modified entity
        new_version = modified_entity.fork()
        
        # Verify forking creates a different ID
        assert new_version.ecs_id != entity.ecs_id
        assert new_version.lineage_id == entity.lineage_id
        assert new_version.parent_id == entity.ecs_id
        
        # Verify data was preserved
        assert new_version.name == "Original"
        assert new_version.value == 20

class TestEntityTracer:
    """Tests for the entity_tracer decorator."""
    
    def test_entity_tracer_decorator(self):
        """Test that the entity_tracer decorator properly tracks and forks entities."""
        entity = SimpleEntity(name="Original", value=10)
        entity.initialize_deps_graph()
        original_id = entity.ecs_id
        
        @entity_tracer
        def modify_entity(e: SimpleEntity) -> SimpleEntity:
            e.value = 20
            return e
        
        # Modify entity through traced function
        result = modify_entity(entity)
        
        # Verify it was forked automatically
        assert result.ecs_id != original_id
        assert result.value == 20
        assert result.parent_id == original_id

class TestDependencyGraph:
    """Tests for entity dependency graph functionality."""
    
    def test_dependency_graph_initialization(self):
        """Test that dependency graphs are properly initialized."""
        # Create a parent with children
        child1 = SimpleEntity(name="Child1", value=1)
        child2 = SimpleEntity(name="Child2", value=2)
        
        parent = ParentEntity(name="Parent", children=[child1, child2])
        
        # Initialize parent's dependency graph
        parent.initialize_deps_graph()
        
        # Verify graph was built
        assert parent.deps_graph is not None
        
        # Verify children have same graph
        assert child1.deps_graph is parent.deps_graph
        assert child2.deps_graph is parent.deps_graph
        
        # Verify graph contains correct relationships
        node = parent.deps_graph.get_node(parent.ecs_id)
        assert node is not None
        
        # Parent depends on children
        assert child1.ecs_id in node.dependencies
        assert child2.ecs_id in node.dependencies
        
        # Children are dependent on parent
        child1_node = parent.deps_graph.get_node(child1.ecs_id)
        child2_node = parent.deps_graph.get_node(child2.ecs_id)
        
        assert child1_node is not None
        assert child2_node is not None
        assert parent.ecs_id in child1_node.dependents
        assert parent.ecs_id in child2_node.dependents