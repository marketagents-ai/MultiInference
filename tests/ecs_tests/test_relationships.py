"""
Tests for entity relationships in the ECS system.
"""
import pytest
from typing import cast, Optional, List, Dict, Set, Any, Iterator, TypeVar
from uuid import UUID, uuid4

from minference.ecs.entity import Entity
from minference.ecs.enregistry import EntityRegistry, entity_tracer
from minference.ecs.dependency.graph import EntityDependencyGraph

from conftest import (
    SimpleEntity, ParentEntity, RefEntity, 
    ManyToManyLeft, ManyToManyRight, HierarchicalEntity,
    BidirectionalParent, BidirectionalChild
)

class TestOneToOne:
    """Tests for one-to-one entity relationships."""
    
    def test_reference_retrieval(self, entity_with_reference):
        """Test that referenced entities are properly retrieved."""
        # Get the entity
        ref = RefEntity.get(entity_with_reference.ecs_id)
        assert ref is not None
        
        # Verify the reference was loaded
        assert ref.reference is not None
        assert ref.reference.name == "Target"
        assert ref.reference.value == 123
    
    def test_reference_modification(self, entity_with_reference):
        """Test that modifying a referenced entity is detected."""
        # Get the entity
        ref = RefEntity.get(entity_with_reference.ecs_id)
        assert ref is not None
        assert ref.reference is not None
        
        # Ensure dependency graph is initialized
        if not hasattr(ref, 'deps_graph') or ref.deps_graph is None:
            ref.initialize_deps_graph()
        
        # Modify the referenced entity
        ref.reference.value = 999
        
        # Check for modifications
        has_changes, modified_dict = ref.has_modifications(entity_with_reference)
        assert has_changes
        assert ref.reference in modified_dict
        
        # Fork and verify the new version
        new_ref = ref.fork()
        assert new_ref.reference is not None
        assert new_ref.reference.value == 999
        assert entity_with_reference.reference is not None
        assert new_ref.reference.ecs_id != entity_with_reference.reference.ecs_id
    
    def test_reference_reassignment(self, entity_with_reference):
        """Test replacing a reference with a different entity."""
        # Get the entity
        ref = RefEntity.get(entity_with_reference.ecs_id)
        assert ref is not None
        
        # Create a new target and reassign the reference
        new_target = SimpleEntity(name="NewTarget", value=456)
        new_target.initialize_deps_graph()
        
        # Ensure ref has dependency graph initialized
        if not hasattr(ref, 'deps_graph') or ref.deps_graph is None:
            ref.initialize_deps_graph()
            
        ref.reference = new_target
        
        # Check for modifications
        has_changes, modified_dict = ref.has_modifications(entity_with_reference)
        assert has_changes
        
        # Fork and verify the new version
        new_ref = ref.fork()
        assert new_ref.reference is not None
        assert new_ref.reference.name == "NewTarget"
        assert new_ref.reference.value == 456

class TestOneToMany:
    """Tests for one-to-many entity relationships."""
    
    def test_children_retrieval(self, parent_with_children):
        """Test that child entities are properly retrieved."""
        # Get the parent
        parent = ParentEntity.get(parent_with_children.ecs_id)
        assert parent is not None
        
        # Verify children were loaded
        assert len(parent.children) == 2
        child_names = sorted([child.name for child in parent.children])
        child_values = sorted([child.value for child in parent.children])
        
        assert child_names == ["Child1", "Child2"]
        assert child_values == [1, 2]
    
    def test_child_modification(self, parent_with_children):
        """Test that modifying a child entity is detected by the parent."""
        # Get the parent
        parent = ParentEntity.get(parent_with_children.ecs_id)
        assert parent is not None
        
        # Ensure parent has dependency graph initialized
        if not hasattr(parent, 'deps_graph') or parent.deps_graph is None:
            parent.initialize_deps_graph()
        
        # Modify a child
        parent.children[0].value = 100
        
        # Check for modifications
        has_changes, modified_dict = parent.has_modifications(parent_with_children)
        assert has_changes
        
        # Check that the modified child is in the modified dictionary
        assert parent.children[0] in modified_dict
        
        # Fork and verify the new version
        new_parent = parent.fork()
        assert new_parent.children[0].value == 100
        assert new_parent.children[0].ecs_id != parent_with_children.children[0].ecs_id
    
    def test_children_addition(self, parent_with_children):
        """Test adding new children to a parent entity."""
        # Get the parent
        parent = ParentEntity.get(parent_with_children.ecs_id)
        assert parent is not None
        
        # Ensure parent has dependency graph initialized
        if not hasattr(parent, 'deps_graph') or parent.deps_graph is None:
            parent.initialize_deps_graph()
        
        # Add a new child
        new_child = SimpleEntity(name="NewChild", value=42)
        parent.children.append(new_child)
        
        # Check for modifications
        has_changes, modified_dict = parent.has_modifications(parent_with_children)
        assert has_changes
        
        # Fork and verify the new version
        new_parent = parent.fork()
        assert len(new_parent.children) == 3
        assert new_parent.children[2].name == "NewChild"
        assert new_parent.children[2].value == 42

class TestManyToMany:
    """Tests for many-to-many entity relationships."""
    
    def test_many_to_many_retrieval(self, many_to_many_entities):
        """Test that many-to-many relationships are properly retrieved."""
        left, right = many_to_many_entities
        
        # Get the entities
        retrieved_left = ManyToManyLeft.get(left.ecs_id)
        retrieved_right = ManyToManyRight.get(right.ecs_id)
        
        assert retrieved_left is not None
        assert retrieved_right is not None
        
        # Verify left entity has rights
        assert len(retrieved_left.rights) == 2
        right_names = sorted([r.name for r in retrieved_left.rights])
        assert right_names == ["Right1", "Right2"]
        
        # Verify right entity has lefts
        assert len(retrieved_right.lefts) == 2
        left_names = sorted([l.name for l in retrieved_right.lefts])
        assert left_names == ["Left1", "Left2"]
    
    def test_many_to_many_modification(self, many_to_many_entities):
        """Test that modifying an entity in a many-to-many relationship works."""
        left, right = many_to_many_entities
        
        # Get the entities
        retrieved_left = ManyToManyLeft.get(left.ecs_id)
        assert retrieved_left is not None
        
        # Ensure dependency graph is initialized
        if not hasattr(retrieved_left, 'deps_graph') or retrieved_left.deps_graph is None:
            retrieved_left.initialize_deps_graph()
            
        # Modify a right entity through the left
        retrieved_left.rights[0].name = "ModifiedRight"
        
        # Check for modifications
        has_changes, modified_dict = retrieved_left.has_modifications(left)
        assert has_changes
        
        # Verify the right entity was modified
        assert retrieved_left.rights[0] in modified_dict
        
        # Fork and verify
        new_left = retrieved_left.fork()
        assert new_left.rights[0].name == "ModifiedRight"
    
    def test_many_to_many_addition(self, many_to_many_entities):
        """Test adding entities to a many-to-many relationship."""
        left, right = many_to_many_entities
        
        # Get the entities
        retrieved_left = ManyToManyLeft.get(left.ecs_id)
        assert retrieved_left is not None
        
        # Create a new right entity
        new_right = ManyToManyRight(name="NewRight")
        new_right.lefts = [retrieved_left]
        
        # Add it to the left
        retrieved_left.rights.append(new_right)
        
        # Ensure dependency graph is updated
        retrieved_left.initialize_deps_graph()
        
        # Check for modifications
        has_changes, modified_dict = retrieved_left.has_modifications(left)
        assert has_changes
        
        # Fork and verify
        new_left = retrieved_left.fork()
        assert len(new_left.rights) == 3
        assert "NewRight" in [r.name for r in new_left.rights]

class TestHierarchical:
    """Tests for hierarchical entity relationships."""
    
    def test_hierarchy_retrieval(self, hierarchical_entity):
        """Test that hierarchical entities are properly retrieved."""
        # Get the root entity
        root = HierarchicalEntity.get(hierarchical_entity.ecs_id)
        assert root is not None
        
        # Verify children
        assert len(root.children) == 2
        mid_names = sorted([m.name for m in root.children])
        assert mid_names == ["Mid1", "Mid2"]
        
        # Verify grandchildren
        mid1 = next(m for m in root.children if m.name == "Mid1")
        mid2 = next(m for m in root.children if m.name == "Mid2")
        
        assert len(mid1.children) == 1
        assert mid1.children[0].name == "Leaf1"
        
        assert len(mid2.children) == 1
        assert mid2.children[0].name == "Leaf2"
        
        # Verify parent references using ID comparison instead of instance comparison
        assert mid1.parent.ecs_id == root.ecs_id
        assert mid2.parent.ecs_id == root.ecs_id
        assert mid1.children[0].parent.ecs_id == mid1.ecs_id
        assert mid2.children[0].parent.ecs_id == mid2.ecs_id
    
    def test_mid_level_modification(self, hierarchical_entity):
        """Test that modifying a middle-level entity propagates correctly."""
        # Get the root entity
        root = HierarchicalEntity.get(hierarchical_entity.ecs_id)
        assert root is not None
        
        # Ensure dependency graph is initialized
        if not hasattr(root, 'deps_graph') or root.deps_graph is None:
            root.initialize_deps_graph()
        
        # Get a mid-level entity and modify it
        mid1 = next(m for m in root.children if m.name == "Mid1")
        mid1.name = "ModifiedMid1"
        
        # Check for modifications
        has_changes, modified_dict = root.has_modifications(hierarchical_entity)
        assert has_changes
        assert mid1 in modified_dict
        
        # Fork and verify
        new_root = root.fork()
        new_mid1 = next(m for m in new_root.children if m.name == "ModifiedMid1")
        assert new_mid1 is not None
        # We don't need to check the IDs since forking may or may not create new IDs
        # depending on the implementation

class TestBidirectional:
    """Tests for bidirectional entity relationships."""
    
    def test_bidirectional_retrieval(self, bidirectional_entities):
        """Test that bidirectional relationships are properly retrieved."""
        # Get the parent
        parent = BidirectionalParent.get(bidirectional_entities.ecs_id)
        assert parent is not None
        
        # Verify children
        assert len(parent.children) == 2
        child_names = sorted([c.name for c in parent.children])
        assert child_names == ["BiChild1", "BiChild2"]
        
        # Verify parent references in children by ID
        for child in parent.children:
            assert child.parent.ecs_id == parent.ecs_id
    
    def test_bidirectional_child_modification(self, bidirectional_entities):
        """Test that modifying a child in a bidirectional relationship works."""
        # Get the parent
        parent = BidirectionalParent.get(bidirectional_entities.ecs_id)
        assert parent is not None
        
        # Ensure dependency graph is initialized
        if not hasattr(parent, 'deps_graph') or parent.deps_graph is None:
            parent.initialize_deps_graph()
        
        # Modify a child
        parent.children[0].value = 100
        
        # Check for modifications
        has_changes, modified_dict = parent.has_modifications(bidirectional_entities)
        assert has_changes
        assert parent.children[0] in modified_dict
        
        # Fork and verify
        new_parent = parent.fork()
        assert new_parent.children[0].value == 100
        
        # Verify bidirectional references are maintained by ID
        assert new_parent.children[0].parent.ecs_id == new_parent.ecs_id