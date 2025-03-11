"""
Tests for SQLAlchemy entity-ORM conversion roundtrip.

This test suite verifies that:
1. Entity objects can be converted to their ORM counterparts
2. ORM objects can be converted back to entity objects
3. The roundtrip conversion preserves identity under our comparison methods
4. Hierarchical relationships are properly maintained in the conversion
"""

# Create a mock EntityRegistry for testing
class EntityRegistry:
    """Mock registry for testing."""
    
    @classmethod
    def get_registry_status(cls):
        """Get status information about the registry."""
        return {"storage": "in_memory"}
    
    @classmethod
    def get_cold_snapshot(cls, entity_id):
        """Mock method, always returns None."""
        return None
    
    @classmethod
    def register(cls, entity):
        """Mock registration method."""
        return entity

# Add EntityRegistry to __main__ for the Entity.has_modifications method
import sys
sys.modules['__main__'].__dict__['EntityRegistry'] = EntityRegistry

import unittest
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Set, cast

from sqlalchemy import create_engine, Column, String, Integer, ForeignKey, DateTime, JSON, Uuid, Table
from sqlalchemy.orm import relationship, sessionmaker, Session, mapped_column, Mapped

# Import required classes and functions from the ecs and threads modules
from minference.ecs.entity import Entity, compare_entity_fields
from minference.ecs.storage import create_association_table, BaseEntitySQL
from minference.ecs.storage import Base, EntityBase

# Create a separate metadata instance for these tests
from sqlalchemy import MetaData
test_metadata = MetaData()

# Create a subclass of Base using our test metadata
from sqlalchemy.orm import declarative_base
TestBase = declarative_base(metadata=test_metadata)

# Create a base class for our test entities using the test metadata
class TestEntityBase(TestBase):
    """Abstract base class for our test entity tables."""
    __abstract__ = True
    
    # Primary database key (auto-incremented integer)
    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Entity versioning fields from the Entity class
    ecs_id = mapped_column(Uuid, nullable=False, index=True, unique=True)
    lineage_id = mapped_column(Uuid, nullable=False, index=True)
    parent_id = mapped_column(Uuid, nullable=True, index=True)
    created_at = mapped_column(DateTime(timezone=True), nullable=False)
    old_ids = mapped_column(JSON, nullable=False, default=list)
    
    # The entity type for polymorphic identity
    entity_type = mapped_column(String(50), nullable=False)
    
    __mapper_args__ = {
        "polymorphic_on": entity_type,
    }

# Create test entities
class SimpleEntity(Entity):
    """A simple entity with basic fields."""
    name: str
    value: int
    
class NestedEntity(Entity):
    """An entity containing another entity."""
    title: str
    nested: Optional['SimpleEntity'] = None
    
class ParentEntity(Entity):
    """An entity with a list of child entities."""
    name: str
    children: List['ChildEntity'] = []
    
class ChildEntity(Entity):
    """A child entity that belongs to a parent."""
    name: str
    data: str

class ComplexEntity(Entity):
    """An entity with multiple types of relationships."""
    name: str
    simple: Optional[SimpleEntity] = None
    nested: Optional[NestedEntity] = None
    children: List[ChildEntity] = []
    
# Create corresponding ORM models
class SimpleEntityORM(TestEntityBase):
    """ORM model for SimpleEntity."""
    __tablename__ = "test_simpleentityorm"
    
    __mapper_args__ = {
        "polymorphic_identity": "simpleentity"
    }
    
    name: Mapped[str] = mapped_column(String(100))
    value: Mapped[int] = mapped_column(Integer)
    
    def to_entity(self) -> SimpleEntity:
        """Convert ORM to Entity."""
        # Convert string UUIDs back to UUID objects
        uuid_old_ids = [uuid.UUID(uid_str) if isinstance(uid_str, str) else uid_str for uid_str in self.old_ids]
        
        return SimpleEntity(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=uuid_old_ids,
            from_storage=True,
            name=self.name,
            value=self.value
        )
    
    @classmethod
    def from_entity(cls, entity: SimpleEntity) -> 'SimpleEntityORM':
        """Convert Entity to ORM."""
        # Convert UUIDs to strings for JSON serialization
        str_old_ids = [str(uid) for uid in entity.old_ids]
        
        return cls(
            ecs_id=entity.ecs_id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            created_at=entity.created_at,
            old_ids=str_old_ids,  # Store as strings instead of UUID objects
            name=entity.name,
            value=entity.value,
            entity_type="simpleentity"
        )

class NestedEntityORM(TestEntityBase):
    """ORM model for NestedEntity."""
    __tablename__ = "test_nestedentityorm"
    
    __mapper_args__ = {
        "polymorphic_identity": "nestedentity"
    }
    
    title: Mapped[str] = mapped_column(String(100))
    nested_id: Mapped[Optional[int]] = mapped_column(ForeignKey("test_simpleentityorm.id"), nullable=True)
    nested_orm: Mapped[Optional[SimpleEntityORM]] = relationship(SimpleEntityORM)
    
    def to_entity(self) -> NestedEntity:
        """Convert ORM to Entity."""
        # Convert string UUIDs back to UUID objects
        uuid_old_ids = [uuid.UUID(uid_str) if isinstance(uid_str, str) else uid_str for uid_str in self.old_ids]
        
        entity = NestedEntity(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=uuid_old_ids,
            from_storage=True,
            title=self.title,
            nested=self.nested_orm.to_entity() if self.nested_orm else None
        )
        return entity
    
    @classmethod
    def from_entity(cls, entity: NestedEntity) -> 'NestedEntityORM':
        """Convert Entity to ORM."""
        # Convert UUIDs to strings for JSON serialization
        str_old_ids = [str(uid) for uid in entity.old_ids]
        
        orm = cls(
            ecs_id=entity.ecs_id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            created_at=entity.created_at,
            old_ids=str_old_ids,  # Store as strings instead of UUID objects
            title=entity.title,
            entity_type="nestedentity"
        )
        return orm
    
    def handle_relationships(self, entity: NestedEntity, session: Session, orm_objects: Dict[uuid.UUID, Any]) -> None:
        """Handle relationships for this entity."""
        if entity.nested:
            if entity.nested.ecs_id in orm_objects:
                self.nested_orm = cast(SimpleEntityORM, orm_objects[entity.nested.ecs_id])
            else:
                # Query for the related entity if not in orm_objects
                self.nested_orm = session.query(SimpleEntityORM).filter(
                    SimpleEntityORM.ecs_id == entity.nested.ecs_id
                ).first()

class ChildEntityORM(TestEntityBase):
    """ORM model for ChildEntity."""
    __tablename__ = "test_childentityorm"
    
    __mapper_args__ = {
        "polymorphic_identity": "childentity"
    }
    
    name: Mapped[str] = mapped_column(String(100))
    data: Mapped[str] = mapped_column(String(255))
    
    def to_entity(self) -> ChildEntity:
        """Convert ORM to Entity."""
        # Convert string UUIDs back to UUID objects
        uuid_old_ids = [uuid.UUID(uid_str) if isinstance(uid_str, str) else uid_str for uid_str in self.old_ids]
        
        return ChildEntity(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=uuid_old_ids,
            from_storage=True,
            name=self.name,
            data=self.data
        )
    
    @classmethod
    def from_entity(cls, entity: ChildEntity) -> 'ChildEntityORM':
        """Convert Entity to ORM."""
        # Convert UUIDs to strings for JSON serialization
        str_old_ids = [str(uid) for uid in entity.old_ids]
        
        return cls(
            ecs_id=entity.ecs_id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            created_at=entity.created_at,
            old_ids=str_old_ids,  # Store as strings instead of UUID objects
            name=entity.name,
            data=entity.data,
            entity_type="childentity"
        )

# Create association table for many-to-many relationship
parent_child_association = Table(
    "test_parent_child_assoc",
    test_metadata,
    Column("parent_id", Integer, ForeignKey("test_parententityorm.id")),
    Column("child_id", Integer, ForeignKey("test_childentityorm.id"))
)

class ParentEntityORM(TestEntityBase):
    """ORM model for ParentEntity."""
    __tablename__ = "test_parententityorm"
    
    __mapper_args__ = {
        "polymorphic_identity": "parententity"
    }
    
    name: Mapped[str] = mapped_column(String(100))
    children_orm: Mapped[List[ChildEntityORM]] = relationship(
        ChildEntityORM, 
        secondary=parent_child_association
    )
    
    def to_entity(self) -> ParentEntity:
        """Convert ORM to Entity."""
        # Convert string UUIDs back to UUID objects
        uuid_old_ids = [uuid.UUID(uid_str) if isinstance(uid_str, str) else uid_str for uid_str in self.old_ids]
        
        return ParentEntity(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=uuid_old_ids,
            from_storage=True,
            name=self.name,
            children=[child.to_entity() for child in self.children_orm]
        )
    
    @classmethod
    def from_entity(cls, entity: ParentEntity) -> 'ParentEntityORM':
        """Convert Entity to ORM."""
        # Convert UUIDs to strings for JSON serialization
        str_old_ids = [str(uid) for uid in entity.old_ids]
        
        return cls(
            ecs_id=entity.ecs_id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            created_at=entity.created_at,
            old_ids=str_old_ids,  # Store as strings instead of UUID objects
            name=entity.name,
            entity_type="parententity"
        )
    
    def handle_relationships(self, entity: ParentEntity, session: Session, orm_objects: Dict[uuid.UUID, Any]) -> None:
        """Handle relationships for this entity."""
        # Clear existing children to avoid duplicates
        self.children_orm = []
        
        # Add each child
        for child in entity.children:
            if child.ecs_id in orm_objects:
                self.children_orm.append(cast(ChildEntityORM, orm_objects[child.ecs_id]))
            else:
                # Query for the related entity if not in orm_objects
                child_orm = session.query(ChildEntityORM).filter(
                    ChildEntityORM.ecs_id == child.ecs_id
                ).first()
                if child_orm:
                    self.children_orm.append(child_orm)

class ComplexEntityORM(TestEntityBase):
    """ORM model for ComplexEntity."""
    __tablename__ = "test_complexentityorm"
    
    __mapper_args__ = {
        "polymorphic_identity": "complexentity"
    }
    
    name: Mapped[str] = mapped_column(String(100))
    
    # One-to-one relationships
    simple_id: Mapped[Optional[int]] = mapped_column(ForeignKey("test_simpleentityorm.id"), nullable=True)
    nested_id: Mapped[Optional[int]] = mapped_column(ForeignKey("test_nestedentityorm.id"), nullable=True)
    
    # Relationship properties
    simple_orm: Mapped[Optional[SimpleEntityORM]] = relationship(SimpleEntityORM)
    nested_orm: Mapped[Optional[NestedEntityORM]] = relationship(NestedEntityORM)
    
    # Create a specific association table for the complex-child relationship
    complex_child_association = Table(
        "test_complex_child_assoc",
        test_metadata,
        Column("complex_id", Integer, ForeignKey("test_complexentityorm.id")),
        Column("child_id", Integer, ForeignKey("test_childentityorm.id"))
    )
    
    children_orm: Mapped[List[ChildEntityORM]] = relationship(
        ChildEntityORM, 
        secondary=complex_child_association
    )
    
    def to_entity(self) -> ComplexEntity:
        """Convert ORM to Entity."""
        # Convert string UUIDs back to UUID objects
        uuid_old_ids = [uuid.UUID(uid_str) if isinstance(uid_str, str) else uid_str for uid_str in self.old_ids]
        
        return ComplexEntity(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=uuid_old_ids,
            from_storage=True,
            name=self.name,
            simple=self.simple_orm.to_entity() if self.simple_orm else None,
            nested=self.nested_orm.to_entity() if self.nested_orm else None,
            children=[child.to_entity() for child in self.children_orm]
        )
    
    @classmethod
    def from_entity(cls, entity: ComplexEntity) -> 'ComplexEntityORM':
        """Convert Entity to ORM."""
        # Convert UUIDs to strings for JSON serialization
        str_old_ids = [str(uid) for uid in entity.old_ids]
        
        return cls(
            ecs_id=entity.ecs_id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            created_at=entity.created_at,
            old_ids=str_old_ids,  # Store as strings instead of UUID objects
            name=entity.name,
            entity_type="complexentity"
        )
    
    def handle_relationships(self, entity: ComplexEntity, session: Session, orm_objects: Dict[uuid.UUID, Any]) -> None:
        """Handle relationships for this entity."""
        # Handle simple relationship
        if entity.simple:
            if entity.simple.ecs_id in orm_objects:
                self.simple_orm = cast(SimpleEntityORM, orm_objects[entity.simple.ecs_id])
            else:
                # Query for the related entity if not in orm_objects
                self.simple_orm = session.query(SimpleEntityORM).filter(
                    SimpleEntityORM.ecs_id == entity.simple.ecs_id
                ).first()
                
        # Handle nested relationship
        if entity.nested:
            if entity.nested.ecs_id in orm_objects:
                self.nested_orm = cast(NestedEntityORM, orm_objects[entity.nested.ecs_id])
            else:
                # Query for the related entity if not in orm_objects
                self.nested_orm = session.query(NestedEntityORM).filter(
                    NestedEntityORM.ecs_id == entity.nested.ecs_id
                ).first()
        
        # Handle children relationship
        self.children_orm = []
        for child in entity.children:
            if child.ecs_id in orm_objects:
                self.children_orm.append(cast(ChildEntityORM, orm_objects[child.ecs_id]))
            else:
                # Query for the related entity if not in orm_objects
                child_orm = session.query(ChildEntityORM).filter(
                    ChildEntityORM.ecs_id == child.ecs_id
                ).first()
                if child_orm:
                    self.children_orm.append(child_orm)

def compare_entities(entity1, entity2):
    """
    Compare two entities directly using the compare_entity_fields function.
    Returns True if they are equal, False otherwise.
    """
    has_diffs, _ = compare_entity_fields(entity1, entity2)
    return not has_diffs

class TestEntityConversion(unittest.TestCase):
    """Test suite for entity-ORM conversion."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create in-memory SQLite database
        self.engine = create_engine("sqlite:///:memory:")
        
        # Create all tables
        TestBase.metadata.create_all(self.engine)
        
        # Create a session
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        
        # Create entity-to-ORM mapping
        self.entity_to_orm_map = {
            SimpleEntity: SimpleEntityORM,
            NestedEntity: NestedEntityORM,
            ChildEntity: ChildEntityORM,
            ParentEntity: ParentEntityORM,
            ComplexEntity: ComplexEntityORM,
            Entity: BaseEntitySQL  # Fallback
        }
        
    def tearDown(self):
        """Clean up after tests."""
        self.session.close()
        TestBase.metadata.drop_all(self.engine)
    
    def test_simple_entity_conversion(self):
        """Test conversion of a simple entity with primitive fields."""
        # Create a simple entity
        entity = SimpleEntity(
            name="Test Entity",
            value=42
        )
        
        # Convert to ORM
        orm = SimpleEntityORM.from_entity(entity)
        
        # Save to database
        self.session.add(orm)
        self.session.commit()
        
        # Retrieve from database
        retrieved_orm = self.session.query(SimpleEntityORM).filter(
            SimpleEntityORM.ecs_id == entity.ecs_id
        ).first()
        
        # Convert back to entity
        retrieved_entity = retrieved_orm.to_entity()
        
        # Check if fields match
        self.assertEqual(entity.name, retrieved_entity.name)
        self.assertEqual(entity.value, retrieved_entity.value)
        self.assertEqual(entity.ecs_id, retrieved_entity.ecs_id)
        
        # Check if entities are equal under our comparison method
        self.assertTrue(compare_entities(entity, retrieved_entity), "Entities should be equal after roundtrip conversion")
    
    def test_nested_entity_conversion(self):
        """Test conversion of an entity containing another entity."""
        # Create nested structure
        nested = SimpleEntity(name="Nested", value=123)
        entity = NestedEntity(title="Parent", nested=nested)
        
        # Convert and store both entities
        nested_orm = SimpleEntityORM.from_entity(nested)
        self.session.add(nested_orm)
        
        entity_orm = NestedEntityORM.from_entity(entity)
        self.session.add(entity_orm)
        
        # Handle relationships
        orm_objects = {nested.ecs_id: nested_orm, entity.ecs_id: entity_orm}
        entity_orm.handle_relationships(entity, self.session, orm_objects)
        
        self.session.commit()
        
        # Retrieve from database with eager loading
        from sqlalchemy.orm import joinedload
        retrieved_orm = self.session.query(NestedEntityORM).options(
            joinedload(NestedEntityORM.nested_orm)
        ).filter(
            NestedEntityORM.ecs_id == entity.ecs_id
        ).first()
        
        # Convert back to entity
        retrieved_entity = retrieved_orm.to_entity()
        
        # Check if fields match
        self.assertEqual(entity.title, retrieved_entity.title)
        self.assertIsNotNone(retrieved_entity.nested)
        self.assertEqual(nested.name, retrieved_entity.nested.name)
        self.assertEqual(nested.value, retrieved_entity.nested.value)
        
        # Check if entities are equal under our comparison method
        self.assertTrue(compare_entities(entity, retrieved_entity), "Entities should be equal after roundtrip conversion")
    
    def test_parent_child_relationship(self):
        """Test conversion of an entity with a list of child entities."""
        # Create parent with children
        children = [
            ChildEntity(name="Child 1", data="Data 1"),
            ChildEntity(name="Child 2", data="Data 2"),
            ChildEntity(name="Child 3", data="Data 3")
        ]
        parent = ParentEntity(name="Parent", children=children)
        
        # Convert and store all entities
        orm_objects = {}
        
        # Store children first
        for child in children:
            child_orm = ChildEntityORM.from_entity(child)
            self.session.add(child_orm)
            orm_objects[child.ecs_id] = child_orm
        
        # Store parent
        parent_orm = ParentEntityORM.from_entity(parent)
        self.session.add(parent_orm)
        orm_objects[parent.ecs_id] = parent_orm
        
        # Handle relationships
        parent_orm.handle_relationships(parent, self.session, orm_objects)
        
        self.session.commit()
        
        # Retrieve from database
        retrieved_orm = self.session.query(ParentEntityORM).filter(
            ParentEntityORM.ecs_id == parent.ecs_id
        ).first()
        
        # Convert back to entity
        retrieved_entity = retrieved_orm.to_entity()
        
        # Check if fields match
        self.assertEqual(parent.name, retrieved_entity.name)
        self.assertEqual(len(parent.children), len(retrieved_entity.children))
        
        # Check if child names are preserved
        retrieved_names = [child.name for child in retrieved_entity.children]
        original_names = [child.name for child in parent.children]
        self.assertCountEqual(original_names, retrieved_names)
        
        # Check if entities are equal under our comparison method
        self.assertTrue(compare_entities(parent, retrieved_entity), "Entities should be equal after roundtrip conversion")
    
    def test_complex_entity_conversion(self):
        """Test conversion of a complex entity with various relationship types."""
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
        
        # Convert and store all entities
        orm_objects = {}
        
        # Store all sub-entities first
        sub_entities = [simple, nested_inner, nested] + children
        for sub in sub_entities:
            if isinstance(sub, SimpleEntity):
                orm = SimpleEntityORM.from_entity(sub)
            elif isinstance(sub, NestedEntity):
                orm = NestedEntityORM.from_entity(sub)
            elif isinstance(sub, ChildEntity):
                orm = ChildEntityORM.from_entity(sub)
            else:
                continue
                
            self.session.add(orm)
            orm_objects[sub.ecs_id] = orm
        
        # Handle nested relationship
        nested_orm = cast(NestedEntityORM, orm_objects[nested.ecs_id])
        nested_orm.handle_relationships(nested, self.session, orm_objects)
        
        # Store complex entity
        complex_orm = ComplexEntityORM.from_entity(complex_entity)
        self.session.add(complex_orm)
        orm_objects[complex_entity.ecs_id] = complex_orm
        
        # Handle complex relationships
        complex_orm.handle_relationships(complex_entity, self.session, orm_objects)
        
        self.session.commit()
        
        # Retrieve from database
        retrieved_orm = self.session.query(ComplexEntityORM).filter(
            ComplexEntityORM.ecs_id == complex_entity.ecs_id
        ).first()
        
        # Convert back to entity
        retrieved_entity = retrieved_orm.to_entity()
        
        # Check if fields match
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
        
        # Check children
        self.assertEqual(len(complex_entity.children), len(retrieved_entity.children))
        
        # Check if entities are equal under our comparison method
        self.assertTrue(compare_entities(complex_entity, retrieved_entity), "Entities should be equal after roundtrip conversion")
    
    def test_multiple_entity_comparison(self):
        """Test comparison of multiple entities after conversion."""
        # Create a complex structure with multiple related entities
        simple = SimpleEntity(name="Simple", value=42)
        nested = NestedEntity(title="Nested", nested=simple)
        
        # Convert to ORM
        simple_orm = SimpleEntityORM.from_entity(simple)
        nested_orm = NestedEntityORM.from_entity(nested)
        
        # Store in database
        self.session.add(simple_orm)
        self.session.add(nested_orm)
        
        # Handle relationships
        orm_objects = {simple.ecs_id: simple_orm, nested.ecs_id: nested_orm}
        nested_orm.handle_relationships(nested, self.session, orm_objects)
        
        self.session.commit()
        
        # Retrieve from database
        retrieved_nested_orm = self.session.query(NestedEntityORM).filter(
            NestedEntityORM.ecs_id == nested.ecs_id
        ).first()
        
        # Convert back to entity
        retrieved_nested = retrieved_nested_orm.to_entity()
        
        # Check full tree comparison
        self.assertTrue(compare_entities(nested, retrieved_nested), "Entities should be equal after roundtrip conversion")
        
        # Modify a field and check again
        modified_nested = retrieved_nested
        modified_nested.title = "Modified Title"
        
        self.assertFalse(compare_entities(nested, modified_nested), "Entities should be different after modification")
        
        # Ensure nested entity comparison works correctly
        if retrieved_nested.nested and nested.nested:
            self.assertTrue(compare_entities(nested.nested, retrieved_nested.nested), "Nested entities should be equal")

if __name__ == "__main__":
    unittest.main()