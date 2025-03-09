"""
Test script to validate dependency graph integration with Entity registration.
Tests various entity structure patterns to ensure proper registration and no unnecessary forks.
"""
import sys
import logging
from uuid import UUID, uuid4
from typing import Optional, List, Dict, Set, Any, TypeVar, cast
from typing_extensions import Annotated

from tests.ecs_tests.simplified_entity import Entity, EntityRegistry, InMemoryEntityStorage
from tests.ecs_tests.entity_dependencies import EntityDependencyGraph, CycleStatus
from pydantic import Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DepsGraphTest")

# Create type variables for entity types
T_SimpleEntity = TypeVar('T_SimpleEntity', bound='SimpleEntity')
T_ParentEntity = TypeVar('T_ParentEntity', bound='ParentEntity')
T_BidirectionalParent = TypeVar('T_BidirectionalParent', bound='BidirectionalParent')
T_BidirectionalChild = TypeVar('T_BidirectionalChild', bound='BidirectionalChild')
T_CircularRefA = TypeVar('T_CircularRefA', bound='CircularRefA')
T_CircularRefB = TypeVar('T_CircularRefB', bound='CircularRefB')
T_CircularRefC = TypeVar('T_CircularRefC', bound='CircularRefC')
T_DiamondTop = TypeVar('T_DiamondTop', bound='DiamondTop')
T_DiamondLeft = TypeVar('T_DiamondLeft', bound='DiamondLeft')
T_DiamondRight = TypeVar('T_DiamondRight', bound='DiamondRight')
T_DiamondBottom = TypeVar('T_DiamondBottom', bound='DiamondBottom')

# Define test entities for various relationship patterns
class SimpleEntity(Entity):
    """A simple entity with basic fields."""
    name: str
    value: int = 0
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional["SimpleEntity"]:
        """Override to return correct type"""
        return cast(Optional["SimpleEntity"], super().get(entity_id))

class ParentEntity(Entity):
    """Entity with one-to-many relationship to SimpleEntity."""
    name: str
    children: List[SimpleEntity] = Field(default_factory=list)
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional["ParentEntity"]:
        """Override to return correct type"""
        return cast(Optional["ParentEntity"], super().get(entity_id))

class BidirectionalParent(Entity):
    """Parent entity with bidirectional references to children."""
    name: str
    children: List["BidirectionalChild"] = Field(default_factory=list)
    
    def add_child(self, child: "BidirectionalChild") -> None:
        """Add a child with bidirectional reference."""
        self.children.append(child)
        if child:
            child.parent = self
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional["BidirectionalParent"]:
        """Override to return correct type"""
        return cast(Optional["BidirectionalParent"], super().get(entity_id))

class BidirectionalChild(Entity):
    """Child entity with reference back to parent."""
    name: str
    value: int = 0
    parent: Optional["BidirectionalParent"] = None
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional["BidirectionalChild"]:
        """Override to return correct type"""
        return cast(Optional["BidirectionalChild"], super().get(entity_id))

class CircularRefA(Entity):
    """First entity in a circular reference chain."""
    name: str
    ref_to_b: Optional["CircularRefB"] = None
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional["CircularRefA"]:
        """Override to return correct type"""
        return cast(Optional["CircularRefA"], super().get(entity_id))

class CircularRefB(Entity):
    """Second entity in a circular reference chain."""
    name: str
    ref_to_c: Optional["CircularRefC"] = None
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional["CircularRefB"]:
        """Override to return correct type"""
        return cast(Optional["CircularRefB"], super().get(entity_id))

class CircularRefC(Entity):
    """Third entity in a circular reference chain."""
    name: str
    ref_to_a: Optional["CircularRefA"] = None
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional["CircularRefC"]:
        """Override to return correct type"""
        return cast(Optional["CircularRefC"], super().get(entity_id))

class DiamondTop(Entity):
    """Top entity in a diamond pattern."""
    name: str
    left: Optional["DiamondLeft"] = None
    right: Optional["DiamondRight"] = None
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional["DiamondTop"]:
        """Override to return correct type"""
        return cast(Optional["DiamondTop"], super().get(entity_id))

class DiamondLeft(Entity):
    """Left entity in a diamond pattern."""
    name: str
    bottom: Optional["DiamondBottom"] = None
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional["DiamondLeft"]:
        """Override to return correct type"""
        return cast(Optional["DiamondLeft"], super().get(entity_id))

class DiamondRight(Entity):
    """Right entity in a diamond pattern."""
    name: str
    bottom: Optional["DiamondBottom"] = None
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional["DiamondRight"]:
        """Override to return correct type"""
        return cast(Optional["DiamondRight"], super().get(entity_id))

class DiamondBottom(Entity):
    """Bottom entity in a diamond pattern."""
    name: str
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional["DiamondBottom"]:
        """Override to return correct type"""
        return cast(Optional["DiamondBottom"], super().get(entity_id))

# Set up storage
storage = InMemoryEntityStorage()
EntityRegistry.use_storage(storage)

def test_simple_registration():
    """Test registering simple entities without relationships."""
    logger.info("=== TESTING SIMPLE REGISTRATION ===")
    
    # Clear storage
    EntityRegistry.clear()
    
    # Create and register entities
    entity1 = SimpleEntity(name="Simple1", value=1)
    entity2 = SimpleEntity(name="Simple2", value=2)
    
    logger.info(f"Created entities: {entity1.ecs_id}, {entity2.ecs_id}")
    
    # Verify both were registered
    simple1 = SimpleEntity.get(entity1.ecs_id)
    simple2 = SimpleEntity.get(entity2.ecs_id)
    
    assert simple1 is not None, "Entity1 was not registered"
    assert simple2 is not None, "Entity2 was not registered"
    
    # Verify no forks occurred
    assert simple1.ecs_id == entity1.ecs_id, "Entity1 was forked unexpectedly"
    assert simple2.ecs_id == entity2.ecs_id, "Entity2 was forked unexpectedly"
    
    logger.info("Simple registration test passed")

def test_parent_child_registration():
    """Test registering parent-child relationships."""
    logger.info("=== TESTING PARENT-CHILD REGISTRATION ===")
    
    # Clear storage
    EntityRegistry.clear()
    
    # Create children first
    child1 = SimpleEntity(name="Child1", value=1)
    child2 = SimpleEntity(name="Child2", value=2)
    
    # Create parent with children
    parent = ParentEntity(name="Parent", children=[child1, child2])
    
    logger.info(f"Created parent: {parent.ecs_id} with children: {child1.ecs_id}, {child2.ecs_id}")
    
    # Verify entities exist in registry
    retrieved_parent = ParentEntity.get(parent.ecs_id)
    assert retrieved_parent is not None, "Parent was not registered"
    
    # Verify children were also registered
    assert len(retrieved_parent.children) == 2, "Children were not registered with parent"
    
    # Get IDs of registered children
    child_ids = {child.ecs_id for child in retrieved_parent.children}
    
    # Verify original child IDs match registered child IDs (no forks)
    assert child1.ecs_id in child_ids, "Child1 was forked unexpectedly"
    assert child2.ecs_id in child_ids, "Child2 was forked unexpectedly"
    
    logger.info("Parent-child registration test passed")
    
def test_bidirectional_registration():
    """Test registering entities with bidirectional relationships."""
    logger.info("=== TESTING BIDIRECTIONAL REGISTRATION ===")
    
    # Clear storage
    EntityRegistry.clear()
    
    # Clear any previous dependency graphs
    EntityRegistry.clear()
    
    # Create parent and register it first
    parent = BidirectionalParent(name="BiParent")
    logger.info(f"Created parent: {parent.ecs_id}")
    
    # Explicitly register parent to get it into storage
    EntityRegistry.register(parent)
    
    # Create children and register them
    child1 = BidirectionalChild(name="BiChild1", value=1)
    EntityRegistry.register(child1)
    
    child2 = BidirectionalChild(name="BiChild2", value=2)
    EntityRegistry.register(child2)
    
    logger.info(f"Created children: {child1.ecs_id}, {child2.ecs_id}")
    
    # Now set up bidirectional references
    # First get the registered versions (warm copies)
    parent_warm = BidirectionalParent.get(parent.ecs_id)
    child1_warm = BidirectionalChild.get(child1.ecs_id)
    child2_warm = BidirectionalChild.get(child2.ecs_id)
    
    # Set up references on warm copies - using null checks
    if parent_warm is not None and child1_warm is not None:
        parent_warm.add_child(child1_warm)
    if parent_warm is not None and child2_warm is not None:
        parent_warm.add_child(child2_warm)
    
    # Register parent again to save the relationships
    parent_result = parent_warm and EntityRegistry.register(parent_warm)
    
    # Safe logging with null check
    if parent_result is not None:
        logger.info(f"Set up bidirectional references for parent {parent_result.ecs_id}")
    
    # Verify entities exist in registry
    # Use parent_result which is the latest version after adding children
    # Add null check for parent_result
    if parent_result is not None:
        retrieved_parent = BidirectionalParent.get(parent_result.ecs_id)
        assert retrieved_parent is not None, "Parent was not registered"
    else:
        # If parent_result is None, we can't continue the test
        assert False, "Failed to register parent with children"
    
    # After the null check above, we know retrieved_parent is not None, 
    # but we need to explicitly scope it for type checking
    # Create a new variable that we know is not None
    parent = retrieved_parent
    
    # Verify children were also registered
    assert len(parent.children) == 2, "Children were not registered with parent"
    
    # Get the IDs of registered children
    if parent.children:
        child_ids = {child.ecs_id for child in parent.children if child is not None}
        logger.info(f"Child IDs in retrieved parent: {child_ids}")
        
        # Verify bidirectional references are maintained
        for child in parent.children:
            if child is not None:
                assert child.parent is not None, "Child's parent reference was lost"
                assert child.parent.ecs_id == parent.ecs_id, "Child has incorrect parent reference"
    
    # Instead of using the original IDs (which may have changed due to forking),
    # we'll find the children by name
    retrieved_children = []
    for child in parent.children:
        if child is not None and hasattr(child, 'name'):
            if child.name == "BiChild1" or child.name == "BiChild2":
                retrieved_children.append(child)
    
    # Verify we found both children
    assert len(retrieved_children) == 2, "Could not find both children by name"
    
    # Sort by name to make assertions deterministic
    retrieved_children.sort(key=lambda c: c.name)
    child1_retrieved = retrieved_children[0]  # BiChild1
    child2_retrieved = retrieved_children[1]  # BiChild2
    
    # Verify attributes
    assert child1_retrieved.name == "BiChild1", "Child1 has incorrect name"
    assert child2_retrieved.name == "BiChild2", "Child2 has incorrect name"
    assert child1_retrieved.value == 1, "Child1 has incorrect value"
    assert child2_retrieved.value == 2, "Child2 has incorrect value"
    
    # Verify parent references
    assert child1_retrieved.parent is not None, "Child1's parent reference was lost"
    assert child2_retrieved.parent is not None, "Child2's parent reference was lost"
    assert child1_retrieved.parent.ecs_id == retrieved_parent.ecs_id, "Child1 has incorrect parent reference"
    assert child2_retrieved.parent.ecs_id == retrieved_parent.ecs_id, "Child2 has incorrect parent reference"
    
    logger.info("Bidirectional registration test passed")
    
def test_circular_registration():
    """Test registering entities with circular references."""
    logger.info("=== TESTING CIRCULAR REGISTRATION ===")
    
    # Clear storage
    EntityRegistry.clear()
    
    # Similar to bidirectional test, register entities one by one
    # Create entities first
    entity_a = CircularRefA(name="EntityA")
    EntityRegistry.register(entity_a)
    
    entity_b = CircularRefB(name="EntityB")
    EntityRegistry.register(entity_b)
    
    entity_c = CircularRefC(name="EntityC")
    EntityRegistry.register(entity_c)
    
    logger.info(f"Created entities: A({entity_a.ecs_id}), B({entity_b.ecs_id}), C({entity_c.ecs_id})")
    
    # Get registered versions
    a_warm = CircularRefA.get(entity_a.ecs_id)
    b_warm = CircularRefB.get(entity_b.ecs_id)
    c_warm = CircularRefC.get(entity_c.ecs_id)
    
    # Set up circular references on warm copies, with null checks
    if a_warm is not None and b_warm is not None:
        a_warm.ref_to_b = b_warm
    if b_warm is not None and c_warm is not None:
        b_warm.ref_to_c = c_warm
    if c_warm is not None and a_warm is not None:
        c_warm.ref_to_a = a_warm
    
    # Register with circular references, using type assertions
    a_result = a_warm and EntityRegistry.register(a_warm)
    b_result = b_warm and EntityRegistry.register(b_warm)
    c_result = c_warm and EntityRegistry.register(c_warm)
    
    # Safe logging with conditional access
    if (a_result is not None and 
        hasattr(a_result, 'ref_to_b') and a_result.ref_to_b is not None and 
        hasattr(a_result.ref_to_b, 'ref_to_c') and a_result.ref_to_b.ref_to_c is not None):
        logger.info(f"Set up circular references: A({a_result.ecs_id}) -> B({a_result.ref_to_b.ecs_id}) -> C({a_result.ref_to_b.ref_to_c.ecs_id}) -> A")
    
    # Verify entities exist in registry with latest IDs
    if a_result is not None:
        retrieved_a = CircularRefA.get(a_result.ecs_id)
        
        assert retrieved_a is not None, "EntityA was not registered"
        
        # Verify circular references are maintained - we'll navigate the entire circle
        # using safe type access pattern for attributes
        
        # Add explicit type assertions to help the type checker
        a_entity: CircularRefA = retrieved_a
        
        # Check if the ref_to_b attribute exists and is not None
        assert hasattr(a_entity, 'ref_to_b'), "A doesn't have ref_to_b attribute"
        assert a_entity.ref_to_b is not None, "A's reference to B was lost"
        
        # We verified ref_to_b is not None and is a CircularRefB, cast to help type checker
        ref_b: CircularRefB = cast(CircularRefB, a_entity.ref_to_b)
        
        # Check if ref_b has ref_to_c and it's not None
        assert hasattr(ref_b, 'ref_to_c'), "B doesn't have ref_to_c attribute"
        assert ref_b.ref_to_c is not None, "B's reference to C was lost"
        
        # We verified ref_to_c is not None, so cast it to the right type
        ref_c: CircularRefC = cast(CircularRefC, ref_b.ref_to_c)
        
        # Check if ref_c has ref_to_a and it's not None
        assert hasattr(ref_c, 'ref_to_a'), "C doesn't have ref_to_a attribute"
        assert ref_c.ref_to_a is not None, "C's reference to A was lost"
        
        # The circle should be complete - C points back to A
        # Cast to final type to help type checker
        ref_a_final: CircularRefA = cast(CircularRefA, ref_c.ref_to_a)
        assert a_entity.ecs_id == ref_a_final.ecs_id, "Circular reference is broken"
    else:
        assert False, "EntityA was not registered or has no ID"
    
    logger.info("Circular registration test passed")
    
def test_diamond_registration():
    """Test registering entities with diamond pattern dependencies."""
    logger.info("=== TESTING DIAMOND PATTERN REGISTRATION ===")
    
    # Clear storage
    EntityRegistry.clear()
    
    # Register each entity individually first
    bottom = DiamondBottom(name="Bottom")
    EntityRegistry.register(bottom)
    
    # Create middle entities referencing bottom
    left = DiamondLeft(name="Left")
    EntityRegistry.register(left)
    
    right = DiamondRight(name="Right")
    EntityRegistry.register(right)
    
    # Create top entity 
    top = DiamondTop(name="Top")
    EntityRegistry.register(top)
    
    logger.info(f"Created diamond entities: Top({top.ecs_id}), Left({left.ecs_id}), Right({right.ecs_id}), Bottom({bottom.ecs_id})")
    
    # Get warm copies
    top_warm = DiamondTop.get(top.ecs_id)
    left_warm = DiamondLeft.get(left.ecs_id)
    right_warm = DiamondRight.get(right.ecs_id)
    bottom_warm = DiamondBottom.get(bottom.ecs_id)
    
    # Set up references with null checks
    if left_warm is not None and bottom_warm is not None:
        left_warm.bottom = bottom_warm
    if right_warm is not None and bottom_warm is not None:
        right_warm.bottom = bottom_warm
    if top_warm is not None and left_warm is not None:
        top_warm.left = left_warm
    if top_warm is not None and right_warm is not None:
        top_warm.right = right_warm
    
    # Register updated entities using safe patterns
    left_result = left_warm and EntityRegistry.register(left_warm)
    right_result = right_warm and EntityRegistry.register(right_warm)
    top_result = top_warm and EntityRegistry.register(top_warm)
    
    # Log with safe access
    if (top_result and left_result and right_result and bottom and 
        hasattr(top_result, 'ecs_id') and hasattr(left_result, 'ecs_id') and 
        hasattr(right_result, 'ecs_id') and hasattr(bottom, 'ecs_id')):
        logger.info(f"Set up diamond pattern: Top({top_result.ecs_id}) -> Left({left_result.ecs_id})/Right({right_result.ecs_id}) -> Bottom({bottom.ecs_id})")
    
    # Verify entities exist in registry with latest IDs
    retrieved_top = top_result and DiamondTop.get(top_result.ecs_id)
    
    assert retrieved_top is not None, "Top entity was not registered"
    assert retrieved_top.left is not None, "Left entity was not registered"
    assert retrieved_top.right is not None, "Right entity was not registered"
    assert retrieved_top.left.bottom is not None, "Bottom entity was not registered via left"
    assert retrieved_top.right.bottom is not None, "Bottom entity was not registered via right"
    
    # Verify the bottom entity is the same instance in both paths
    if (retrieved_top.left and retrieved_top.right and 
        retrieved_top.left.bottom and retrieved_top.right.bottom):
        assert retrieved_top.left.bottom.ecs_id == retrieved_top.right.bottom.ecs_id, "Bottom entity is not shared"
    
    logger.info("Diamond pattern registration test passed")
    
def test_modification_detection():
    """Test modification detection and forking with dependencies."""
    logger.info("=== TESTING MODIFICATION DETECTION ===")
    
    # Clear storage
    EntityRegistry.clear()
    
    # Similar to bidirectional test, register entities one by one
    # Create parent and register it first
    parent = BidirectionalParent(name="BiParent")
    parent_result = EntityRegistry.register(parent)
    logger.info(f"Created parent: {parent_result.ecs_id}")
    
    # Create children and register them
    child1 = BidirectionalChild(name="BiChild1", value=1)
    child1_result = EntityRegistry.register(child1)
    
    child2 = BidirectionalChild(name="BiChild2", value=2)
    child2_result = EntityRegistry.register(child2)
    
    logger.info(f"Created children: {child1_result.ecs_id}, {child2_result.ecs_id}")
    
    # Get warm copies
    parent_warm = BidirectionalParent.get(parent_result.ecs_id)
    child1_warm = BidirectionalChild.get(child1_result.ecs_id)
    child2_warm = BidirectionalChild.get(child2_result.ecs_id)
    
    # Set up references with null checks
    if parent_warm is not None and child1_warm is not None:
        parent_warm.add_child(child1_warm)
    if parent_warm is not None and child2_warm is not None:
        parent_warm.add_child(child2_warm)
    
    # Register with relationships
    final_parent = parent_warm and EntityRegistry.register(parent_warm)
    if final_parent is not None:
        logger.info(f"Set up bidirectional relationships: {final_parent.ecs_id}")
    
    # Retrieve registered entities with null check
    if final_parent is not None:
        retrieved_parent = BidirectionalParent.get(final_parent.ecs_id)
        
        # Verify we have children before modifying
        assert retrieved_parent is not None, "Retrieved parent is None"
        assert len(retrieved_parent.children) == 2, "Children were not registered with parent"
    else:
        assert False, "Failed to register parent with relationships"
    
    # We know retrieved_parent is not None after the above check, so we can create 
    # a new variable to help type checker
    parent = retrieved_parent
    
    # Sort children by name to ensure deterministic test
    parent.children.sort(key=lambda c: c.name)
    
    # Modify the first child - we need to verify it exists first
    if parent.children and len(parent.children) > 0 and parent.children[0] is not None:
        parent.children[0].value = 100
        logger.info(f"Modified child value: {parent.children[0].value}")
    else:
        assert False, "First child is missing or None"
    
    # Check modifications
    stored_parent = EntityRegistry.get_cold_snapshot(parent.ecs_id)
    if stored_parent is not None:
        has_changes, modified_dict = parent.has_modifications(stored_parent)
        
        assert has_changes, "Modification was not detected"
        
        # We expect at least the child to be modified
        child_modified = any(isinstance(e, BidirectionalChild) for e in modified_dict.keys())
        assert child_modified, "Child was not marked as modified"
        
        # Store parent ID before forking
        original_id = parent.ecs_id
        
        # Fork the parent
        new_parent = parent.fork()
        logger.info(f"Forked parent: {original_id} -> {new_parent.ecs_id}")
    else:
        assert False, "Stored parent snapshot is None"
    
    # Verify forking occurred correctly 
    # The parent object actually changes its IDs during fork, so we need to compare with original ID
    # We know new_parent is not None from the previous section
    assert new_parent.ecs_id != original_id, "Parent was not forked"
    
    # Verify children exist
    assert new_parent.children, "Children list is empty"
    
    # Sort children to ensure deterministic order
    new_parent.children.sort(key=lambda c: c.name if c is not None else "")
    
    # Verify first child exists and has expected value
    if len(new_parent.children) > 0 and new_parent.children[0] is not None:
        assert new_parent.children[0].value == 100, "Child modification was not preserved"
    else:
        assert False, "First child is missing in forked entity"
    
    # Check bidirectional references in forked entity
    for child in new_parent.children:
        if child is not None:
            assert child.parent is not None, "Child's parent reference was lost"
            assert child.parent.ecs_id == new_parent.ecs_id, "Child's parent reference was not updated to new parent"
    
    logger.info("Modification detection test passed")

def main():
    """Run all tests."""
    # Make EntityRegistry available in __main__
    if '__main__' in sys.modules:
        # Check if the attribute can be set
        if not hasattr(sys.modules['__main__'], 'EntityRegistry'):
            # Use setattr to avoid linting errors
            setattr(sys.modules['__main__'], 'EntityRegistry', EntityRegistry)
    
    # Run tests
    test_simple_registration()
    print("\n")
    test_parent_child_registration()
    print("\n")
    test_bidirectional_registration()
    print("\n")
    test_circular_registration()
    print("\n")
    test_diamond_registration()
    print("\n")
    test_modification_detection()
    
    logger.info("All tests passed!")

if __name__ == "__main__":
    main()