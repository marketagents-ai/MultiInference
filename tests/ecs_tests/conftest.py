"""
Common fixtures and setup for ECS tests.
Provides test entity classes and common setup/teardown functionality.
"""
import pytest
import sys
from uuid import UUID, uuid4
from typing import List, Dict, Optional, Set, Any, Tuple, cast, TypeVar
from pydantic import Field

# Import directly from the main module - we're testing the actual implementation
from minference.ecs.entity import Entity, EntityRegistry, InMemoryEntityStorage, entity_tracer
from minference.ecs.dependency.graph import EntityDependencyGraph, CycleStatus

# Type variables for better type hinting
T_SimpleEntity = TypeVar('T_SimpleEntity', bound='SimpleEntity')
T_ParentEntity = TypeVar('T_ParentEntity', bound='ParentEntity')
T_RefEntity = TypeVar('T_RefEntity', bound='RefEntity')
T_ManyToManyLeft = TypeVar('T_ManyToManyLeft', bound='ManyToManyLeft')
T_ManyToManyRight = TypeVar('T_ManyToManyRight', bound='ManyToManyRight')
T_HierarchicalEntity = TypeVar('T_HierarchicalEntity', bound='HierarchicalEntity')
T_BiParent = TypeVar('T_BiParent', bound='BidirectionalParent')
T_BiChild = TypeVar('T_BiChild', bound='BidirectionalChild')

# ========================================================================
# Test entity classes
# ========================================================================

class SimpleEntity(Entity):
    """A simple entity with basic fields for testing."""
    name: str
    value: int = 0
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional[T_SimpleEntity]:
        """Override to return correct type"""
        entity = super().get(entity_id)
        return cast(Optional[T_SimpleEntity], entity)

class ParentEntity(Entity):
    """Entity with one-to-many relationship to SimpleEntity."""
    name: str
    children: List[SimpleEntity] = Field(default_factory=list)
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional[T_ParentEntity]:
        """Override to return correct type"""
        entity = super().get(entity_id)
        return cast(Optional[T_ParentEntity], entity)

class RefEntity(Entity):
    """Entity with one-to-one reference to another entity."""
    name: str
    reference: Optional[SimpleEntity] = None
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional[T_RefEntity]:
        """Override to return correct type"""
        entity = super().get(entity_id)
        return cast(Optional[T_RefEntity], entity)

class ManyToManyRight(Entity):
    """Right side of a many-to-many relationship."""
    name: str
    lefts: List["ManyToManyLeft"] = Field(default_factory=list)
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional[T_ManyToManyRight]:
        """Override to return correct type"""
        entity = super().get(entity_id)
        return cast(Optional[T_ManyToManyRight], entity)

class ManyToManyLeft(Entity):
    """Left side of a many-to-many relationship."""
    name: str
    rights: List[ManyToManyRight] = Field(default_factory=list)
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional[T_ManyToManyLeft]:
        """Override to return correct type"""
        entity = super().get(entity_id)
        return cast(Optional[T_ManyToManyLeft], entity)

class HierarchicalEntity(Entity):
    """Entity for testing deep hierarchies."""
    name: str
    parent: Optional["HierarchicalEntity"] = None
    children: List["HierarchicalEntity"] = Field(default_factory=list)
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional[T_HierarchicalEntity]:
        """Override to return correct type"""
        entity = super().get(entity_id)
        return cast(Optional[T_HierarchicalEntity], entity)

class BidirectionalChild(Entity):
    """Child entity with reference back to parent."""
    name: str
    value: int = 0
    parent: Optional["BidirectionalParent"] = None
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional[T_BiChild]:
        """Override to return correct type"""
        entity = super().get(entity_id)
        return cast(Optional[T_BiChild], entity)

class BidirectionalParent(Entity):
    """Parent entity with bidirectional references to children."""
    name: str
    children: List[BidirectionalChild] = Field(default_factory=list)
    
    def add_child(self, child: BidirectionalChild) -> None:
        """Add a child with bidirectional reference."""
        if child:
            self.children.append(child)
            child.parent = self
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional[T_BiParent]:
        """Override to return correct type"""
        entity = super().get(entity_id)
        return cast(Optional[T_BiParent], entity)
    
# ========================================================================
# Fixtures
# ========================================================================

@pytest.fixture(autouse=True)
def setup_registry():
    """Setup and teardown for each test."""
    # Make EntityRegistry available in __main__ for Entity methods
    sys.modules['__main__'].EntityRegistry = EntityRegistry
    
    # Use in-memory storage
    storage = InMemoryEntityStorage()
    EntityRegistry.use_storage(storage)
    
    # Run the test
    yield
    
    # Clean up after test
    EntityRegistry.clear()

@pytest.fixture
def simple_entity() -> SimpleEntity:
    """Create a simple test entity."""
    entity = SimpleEntity(name="Test", value=42)
    entity.initialize_deps_graph()
    return entity

@pytest.fixture
def entity_with_reference() -> RefEntity:
    """Create an entity with a reference to another entity."""
    target = SimpleEntity(name="Target", value=123)
    target.initialize_deps_graph()
    
    ref = RefEntity(name="Reference", reference=target)
    ref.initialize_deps_graph()
    return ref

@pytest.fixture
def parent_with_children() -> ParentEntity:
    """Create a parent entity with child entities."""
    child1 = SimpleEntity(name="Child1", value=1)
    child2 = SimpleEntity(name="Child2", value=2)
    
    parent = ParentEntity(name="Parent", children=[child1, child2])
    parent.initialize_deps_graph()
    return parent

@pytest.fixture
def many_to_many_entities() -> tuple[ManyToManyLeft, ManyToManyRight]:
    """Create entities with many-to-many relationships."""
    left1 = ManyToManyLeft(name="Left1")
    left2 = ManyToManyLeft(name="Left2")
    
    right1 = ManyToManyRight(name="Right1")
    right2 = ManyToManyRight(name="Right2")
    
    # Create relationships
    left1.rights = [right1, right2]
    left2.rights = [right1]
    
    right1.lefts = [left1, left2]
    right2.lefts = [left1]
    
    # Initialize dependency graphs
    left1.initialize_deps_graph()
    left2.initialize_deps_graph()
    
    # Explicitly register entities to ensure they're stored properly
    EntityRegistry.register(left1)
    EntityRegistry.register(right1)
    
    return left1, right1

@pytest.fixture
def hierarchical_entity() -> HierarchicalEntity:
    """Create a hierarchical entity structure."""
    # Create a three-level hierarchy
    root = HierarchicalEntity(name="Root")
    mid1 = HierarchicalEntity(name="Mid1", parent=root)
    mid2 = HierarchicalEntity(name="Mid2", parent=root)
    leaf1 = HierarchicalEntity(name="Leaf1", parent=mid1)
    leaf2 = HierarchicalEntity(name="Leaf2", parent=mid2)
    
    # Update children lists
    root.children = [mid1, mid2]
    mid1.children = [leaf1]
    mid2.children = [leaf2]
    
    # Initialize dependency graph
    root.initialize_deps_graph()
    
    # Explicitly register
    EntityRegistry.register(root)
    
    return root

@pytest.fixture
def bidirectional_entities() -> BidirectionalParent:
    """Create a parent with bidirectional references to children."""
    parent = BidirectionalParent(name="BiParent")
    child1 = BidirectionalChild(name="BiChild1", value=1)
    child2 = BidirectionalChild(name="BiChild2", value=2)
    
    # Setup bidirectional references
    parent.add_child(child1)
    parent.add_child(child2)
    
    # Initialize dependency graph
    parent.initialize_deps_graph()
    
    # Explicitly register
    EntityRegistry.register(parent)
    
    return parent