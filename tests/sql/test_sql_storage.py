"""
Tests for SQLAlchemy entity storage backend.

This test suite verifies that:
1. SqlEntityStorage correctly stores, retrieves, and manages entities
2. Entity hierarchies and relationships are maintained properly
3. The storage works with various entity types and relationship patterns
4. Version history and lineage tracking works as expected
"""

import unittest
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Type, Optional, Any, Set, cast
from copy import deepcopy

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Add the project root to sys.path for relative imports in tests
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the test entities and ORM models
from tests.sql.test_sql_entity_conversion import (
    SimpleEntity, SimpleEntityORM,
    NestedEntity, NestedEntityORM,
    ParentEntity, ParentEntityORM, 
    ChildEntity, ChildEntityORM,
    ComplexEntity, ComplexEntityORM,
    TestBase as Base, compare_entities
)

# Import the entity storage implementation
from minference.ecs.entity import Entity, EntityRegistry
from tests.sql.sql_entity import (
    EntityBase, BaseEntitySQL, SqlEntityStorage,
    create_association_table
)

# Add EntityRegistry to __main__ for entity methods
import sys
sys.modules['__main__'].__dict__['EntityRegistry'] = EntityRegistry

class TestSqlEntityStorage(unittest.TestCase):
    """Test suite for SqlEntityStorage."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create in-memory SQLite database
        self.engine = create_engine("sqlite:///:memory:")
        
        # Create all tables
        Base.metadata.create_all(self.engine)
        
        # Create a session factory
        self.Session = sessionmaker(bind=self.engine)
        
        # Create entity-to-ORM mapping
        self.entity_to_orm_map = {
            SimpleEntity: SimpleEntityORM,
            NestedEntity: NestedEntityORM,
            ChildEntity: ChildEntityORM,
            ParentEntity: ParentEntityORM,
            ComplexEntity: ComplexEntityORM,
            Entity: BaseEntitySQL  # Fallback
        }
        
        # Create SqlEntityStorage
        self.storage = SqlEntityStorage(
            session_factory=self.Session,
            entity_to_orm_map=self.entity_to_orm_map
        )
        
        # Set up EntityRegistry to use our storage
        self._original_entity_registry = EntityRegistry._storage
        EntityRegistry._storage = self.storage
        
    def tearDown(self):
        """Clean up after tests."""
        # Restore original storage
        EntityRegistry._storage = self._original_entity_registry
        
        # Close engine and drop tables
        Base.metadata.drop_all(self.engine)
        
    def test_simple_entity_storage(self):
        """Test storing and retrieving a simple entity."""
        # Create a simple entity
        entity = SimpleEntity(
            name="Test Entity",
            value=42
        )
        
        # Store in database
        stored_entity = self.storage.register(entity)
        self.assertIsNotNone(stored_entity)
        
        # Should be able to check entity exists
        self.assertTrue(self.storage.has_entity(entity.ecs_id))
        
        # Retrieve from database
        retrieved_entity = self.storage.get(entity.ecs_id)
        self.assertIsNotNone(retrieved_entity)
        
        # Should have the same content but different live_id
        self.assertEqual(entity.ecs_id, retrieved_entity.ecs_id)
        self.assertNotEqual(entity.live_id, retrieved_entity.live_id)
        self.assertEqual(entity.name, retrieved_entity.name)
        self.assertEqual(entity.value, retrieved_entity.value)
        
        # Should be equal under our entity comparison
        self.assertTrue(compare_entities(entity, retrieved_entity))
        
    def test_nested_entity_storage(self):
        """Test storing and retrieving nested entities."""
        # Create nested structure
        nested = SimpleEntity(name="Nested", value=123)
        entity = NestedEntity(title="Parent", nested=nested)
        
        # Store in database - should store both entities
        stored_entity = self.storage.register(entity)
        self.assertIsNotNone(stored_entity)
        
        # Both entities should exist
        self.assertTrue(self.storage.has_entity(entity.ecs_id))
        self.assertTrue(self.storage.has_entity(nested.ecs_id))
        
        # Retrieve from database
        retrieved_entity = self.storage.get(entity.ecs_id)
        self.assertIsNotNone(retrieved_entity)
        self.assertIsNotNone(retrieved_entity.nested)
        
        # Verify nested entity matches
        self.assertEqual(nested.name, retrieved_entity.nested.name)
        self.assertEqual(nested.value, retrieved_entity.nested.value)
        
        # Should be equal under our entity comparison
        self.assertTrue(compare_entities(entity, retrieved_entity))
        
    def test_parent_child_relationship(self):
        """Test storing and retrieving entities with many-to-many relationships."""
        # Create parent with children
        children = [
            ChildEntity(name="Child 1", data="Data 1"),
            ChildEntity(name="Child 2", data="Data 2"),
            ChildEntity(name="Child 3", data="Data 3")
        ]
        parent = ParentEntity(name="Parent", children=children)
        
        # Store in database - should store parent and all children
        stored_entity = self.storage.register(parent)
        self.assertIsNotNone(stored_entity)
        
        # All entities should exist
        self.assertTrue(self.storage.has_entity(parent.ecs_id))
        for child in children:
            self.assertTrue(self.storage.has_entity(child.ecs_id))
        
        # Retrieve from database
        retrieved_entity = self.storage.get(parent.ecs_id)
        self.assertIsNotNone(retrieved_entity)
        
        # Check retrieved entity structure
        self.assertEqual(parent.name, retrieved_entity.name)
        self.assertEqual(len(children), len(retrieved_entity.children))
        
        # Check all children are present
        retrieved_names = [child.name for child in retrieved_entity.children]
        original_names = [child.name for child in children]
        self.assertCountEqual(original_names, retrieved_names)
        
        # Should be equal under our entity comparison
        self.assertTrue(compare_entities(parent, retrieved_entity))
    
    def test_complex_entity_storage(self):
        """Test storing and retrieving a complex entity structure."""
        # Create a complex entity structure
        simple = SimpleEntity(name="Simple", value=42)
        nested_inner = SimpleEntity(name="Inner", value=100)
        nested = NestedEntity(title="Nested", nested=nested_inner)
        children = [
            ChildEntity(name="Child 1", data="Data 1"),
            ChildEntity(name="Child 2", data="Data 2")
        ]
        
        complex_entity = ComplexEntity(
            name="Complex",
            simple=simple,
            nested=nested,
            children=children
        )
        
        # Store in database - should store all entities
        stored_entity = self.storage.register(complex_entity)
        self.assertIsNotNone(stored_entity)
        
        # All entities should exist
        self.assertTrue(self.storage.has_entity(complex_entity.ecs_id))
        self.assertTrue(self.storage.has_entity(simple.ecs_id))
        self.assertTrue(self.storage.has_entity(nested.ecs_id))
        self.assertTrue(self.storage.has_entity(nested_inner.ecs_id))
        for child in children:
            self.assertTrue(self.storage.has_entity(child.ecs_id))
        
        # Retrieve from database
        retrieved_entity = self.storage.get(complex_entity.ecs_id)
        self.assertIsNotNone(retrieved_entity)
        
        # Check complex structure
        self.assertEqual(complex_entity.name, retrieved_entity.name)
        
        # Check simple relationship
        self.assertIsNotNone(retrieved_entity.simple)
        self.assertEqual(simple.name, retrieved_entity.simple.name)
        self.assertEqual(simple.value, retrieved_entity.simple.value)
        
        # Check nested relationship
        self.assertIsNotNone(retrieved_entity.nested)
        self.assertEqual(nested.title, retrieved_entity.nested.title)
        self.assertIsNotNone(retrieved_entity.nested.nested)
        self.assertEqual(nested_inner.name, retrieved_entity.nested.nested.name)
        
        # Check children collection
        self.assertEqual(len(children), len(retrieved_entity.children))
        
        # Should be equal under our entity comparison
        self.assertTrue(compare_entities(complex_entity, retrieved_entity))
    
    def test_entity_modification_and_versioning(self):
        """Test entity modification detection and version creation."""
        # Create an entity with nested structure
        nested = SimpleEntity(name="Original", value=100)
        entity = NestedEntity(title="Parent", nested=nested)
        
        # Store original version
        self.storage.register(entity)
        original_id = entity.ecs_id
        
        # Modify the entity
        entity.title = "Modified Parent"
        
        # Store modified version - should create a new version
        modified_entity = self.storage.register(entity)
        
        # Should have a new ID
        self.assertNotEqual(original_id, modified_entity.ecs_id)
        self.assertEqual(original_id, modified_entity.parent_id)
        
        # Original version should still exist
        original_from_db = self.storage.get(original_id)
        self.assertIsNotNone(original_from_db)
        self.assertEqual("Parent", original_from_db.title)
        
        # Modified version should have the new title
        modified_from_db = self.storage.get(modified_entity.ecs_id)
        self.assertIsNotNone(modified_from_db)
        self.assertEqual("Modified Parent", modified_from_db.title)
        
        # Both should be in the same lineage
        self.assertEqual(original_from_db.lineage_id, modified_from_db.lineage_id)
        
        # Get all versions by lineage
        lineage_entities = self.storage.get_lineage_entities(entity.lineage_id)
        self.assertEqual(2, len(lineage_entities))
        
    def test_get_many_by_id(self):
        """Test getting multiple entities by ID."""
        # Create some entities
        entities = [
            SimpleEntity(name=f"Entity {i}", value=i) 
            for i in range(5)
        ]
        
        # Store all entities
        for entity in entities:
            self.storage.register(entity)
            
        # Get multiple entities by ID
        entity_ids = [entity.ecs_id for entity in entities[:3]]
        retrieved = self.storage.get_many(entity_ids)
        
        # Should get the right number of entities
        self.assertEqual(3, len(retrieved))
        
        # Should get the right entities
        retrieved_names = {entity.name for entity in retrieved}
        expected_names = {entities[i].name for i in range(3)}
        self.assertEqual(expected_names, retrieved_names)
        
    def test_list_by_type(self):
        """Test listing entities by type."""
        # Create entities of different types
        simple1 = SimpleEntity(name="Simple 1", value=1)
        simple2 = SimpleEntity(name="Simple 2", value=2)
        nested = NestedEntity(title="Nested", nested=simple1)
        child = ChildEntity(name="Child", data="Data")
        
        # Store all entities
        for entity in [simple1, simple2, nested, child]:
            self.storage.register(entity)
            
        # List all SimpleEntity objects
        simple_list = self.storage.list_by_type(SimpleEntity)
        self.assertEqual(2, len(simple_list))
        
        # List all NestedEntity objects
        nested_list = self.storage.list_by_type(NestedEntity)
        self.assertEqual(1, len(nested_list))
        
        # List all ChildEntity objects
        child_list = self.storage.list_by_type(ChildEntity)
        self.assertEqual(1, len(child_list))

if __name__ == "__main__":
    unittest.main()