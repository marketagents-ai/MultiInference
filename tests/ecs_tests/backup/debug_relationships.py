"""
Debug script to focus on entity relationships and the circular reference problem.
"""
import sys
import logging
from uuid import UUID, uuid4
from typing import Optional, List, Dict, Set, Any, TypeVar, cast

from simplified_entity import Entity, EntityRegistry, InMemoryEntityStorage
from pydantic import Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RelationshipDebug")

# Define type variables for better type checking
T_SimpleEntity = TypeVar('T_SimpleEntity', bound='SimpleEntity')
T_ParentEntity = TypeVar('T_ParentEntity', bound='ParentEntity')
T_ManyToManyLeft = TypeVar('T_ManyToManyLeft', bound='ManyToManyLeft')
T_ManyToManyRight = TypeVar('T_ManyToManyRight', bound='ManyToManyRight')
T_HierarchicalEntity = TypeVar('T_HierarchicalEntity', bound='HierarchicalEntity')

# Define test entities
class SimpleEntity(Entity):
    """A simple entity with basic fields."""
    name: str
    value: int = 0
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional[T_SimpleEntity]:
        """Get entity with proper type casting."""
        entity = super().get(entity_id)
        return cast(Optional[T_SimpleEntity], entity)

class ParentEntity(Entity):
    """Entity with one-to-many relationship to SimpleEntity."""
    name: str
    children: List[SimpleEntity] = Field(default_factory=list)
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional[T_ParentEntity]:
        """Get entity with proper type casting."""
        entity = super().get(entity_id)
        return cast(Optional[T_ParentEntity], entity)

class ManyToManyRight(Entity):
    """Entity for testing many-to-many relationships."""
    name: str
    # The key insight: we need to use IDs instead of direct references for circular references
    left_ids: List[UUID] = Field(default_factory=list)
    # When we need the actual objects, we can look them up by ID
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional[T_ManyToManyRight]:
        """Get entity with proper type casting."""
        entity = super().get(entity_id)
        return cast(Optional[T_ManyToManyRight], entity)
    
    @property
    def lefts(self) -> List["ManyToManyLeft"]:
        """Get the left entities by ID."""
        result: List[ManyToManyLeft] = []
        for left_id in self.left_ids:
            if left_id:
                left = ManyToManyLeft.get(left_id)
                if left:
                    result.append(left)
        return result

class ManyToManyLeft(Entity):
    """Entity for testing many-to-many relationships."""
    name: str
    # Same approach: store IDs instead of direct references
    right_ids: List[UUID] = Field(default_factory=list)
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional[T_ManyToManyLeft]:
        """Get entity with proper type casting."""
        entity = super().get(entity_id)
        return cast(Optional[T_ManyToManyLeft], entity)
    
    @property
    def rights(self) -> List[ManyToManyRight]:
        """Get the right entities by ID."""
        result: List[ManyToManyRight] = []
        for right_id in self.right_ids:
            if right_id:
                right = ManyToManyRight.get(right_id)
                if right:
                    result.append(right)
        return result

class HierarchicalEntity(Entity):
    """Entity for testing deep hierarchies."""
    name: str
    # Store parent by ID to avoid circular references
    parent_id: Optional[UUID] = None
    # Store children by ID
    child_ids: List[UUID] = Field(default_factory=list)
    
    @classmethod
    def get(cls, entity_id: UUID) -> Optional[T_HierarchicalEntity]:
        """Get entity with proper type casting."""
        entity = super().get(entity_id)
        return cast(Optional[T_HierarchicalEntity], entity)
    
    @property
    def parent(self) -> Optional["HierarchicalEntity"]:
        """Get the parent entity."""
        if self.parent_id:
            return HierarchicalEntity.get(self.parent_id)
        return None
    
    @property
    def children(self) -> List["HierarchicalEntity"]:
        """Get child entities."""
        result: List[HierarchicalEntity] = []
        for child_id in self.child_ids:
            if child_id:
                child = HierarchicalEntity.get(child_id)
                if child:
                    result.append(child)
        return result

# Set up storage
storage = InMemoryEntityStorage()
EntityRegistry.use_storage(storage)

def test_one_to_many_relationship():
    """Test one-to-many relationships."""
    logger.info("=== TESTING ONE-TO-MANY RELATIONSHIPS ===")
    
    # Create children
    child1 = SimpleEntity(name="Child1", value=1)
    child2 = SimpleEntity(name="Child2", value=2)
    logger.info(f"Created children: {child1.ecs_id}, {child2.ecs_id}")
    
    # Create parent
    parent = ParentEntity(name="Parent", children=[child1, child2])
    logger.info(f"Created parent: {parent.ecs_id}")
    
    # Register all entities (parent should register children too)
    parent_result = EntityRegistry.register(parent)
    if parent_result:
        logger.info(f"Parent result after registration: {parent_result.ecs_id}")
    
        # Retrieve parent and check children
        retrieved_parent = ParentEntity.get(parent_result.ecs_id)
        if retrieved_parent:
            logger.info(f"Retrieved parent: {retrieved_parent.ecs_id}, children: {len(retrieved_parent.children)}")
            for i, child in enumerate(retrieved_parent.children):
                logger.info(f"  Child {i}: {child.name}, {child.value}, {child.ecs_id}")
        else:
            logger.error("Failed to retrieve parent")
            
        # Modify child and check detection
        if retrieved_parent and retrieved_parent.children and len(retrieved_parent.children) > 0:
            retrieved_parent.children[0].value = 100
            logger.info(f"Modified child 0 value to 100")
            
            # Compare with stored version
            stored = EntityRegistry.get_cold_snapshot(retrieved_parent.ecs_id)
            if stored:
                has_changes, modified = retrieved_parent.has_modifications(stored)
                logger.info(f"Has changes: {has_changes}, modified entities: {len(modified)}")
                
                # Fork if needed
                if has_changes:
                    new_parent = retrieved_parent.fork()
                    logger.info(f"Forked to new parent: {new_parent.ecs_id}")
                    logger.info(f"New parent's children: {len(new_parent.children)}")
                    for i, child in enumerate(new_parent.children):
                        logger.info(f"  Child {i}: {child.name}, {child.value}, {child.ecs_id}")

def test_many_to_many_relationship():
    """Test many-to-many relationships with ID references."""
    logger.info("=== TESTING MANY-TO-MANY RELATIONSHIPS ===")
    
    # Create entities first
    left1 = ManyToManyLeft(name="Left1")
    left2 = ManyToManyLeft(name="Left2")
    right1 = ManyToManyRight(name="Right1")
    right2 = ManyToManyRight(name="Right2")
    
    # Register the entities first so they have IDs
    left1_result = EntityRegistry.register(left1)
    left2_result = EntityRegistry.register(left2)
    right1_result = EntityRegistry.register(right1)
    right2_result = EntityRegistry.register(right2)
    
    # Check that all registrations succeeded
    if not all([left1_result, left2_result, right1_result, right2_result]):
        logger.error("Failed to register some entities")
        return
        
    # Use the registered versions
    left1 = cast(ManyToManyLeft, left1_result)
    left2 = cast(ManyToManyLeft, left2_result)
    right1 = cast(ManyToManyRight, right1_result)
    right2 = cast(ManyToManyRight, right2_result)
    
    logger.info(f"Created left1: {left1.ecs_id}, left2: {left2.ecs_id}")
    logger.info(f"Created right1: {right1.ecs_id}, right2: {right2.ecs_id}")
    
    # Set up relationships using IDs
    left1.right_ids = [right1.ecs_id, right2.ecs_id]
    left2.right_ids = [right1.ecs_id]
    right1.left_ids = [left1.ecs_id, left2.ecs_id]
    right2.left_ids = [left1.ecs_id]
    
    # Update in registry
    left1_updated = EntityRegistry.register(left1)
    left2_updated = EntityRegistry.register(left2)
    right1_updated = EntityRegistry.register(right1)
    right2_updated = EntityRegistry.register(right2)
    
    # Check updates succeeded
    if not all([left1_updated, left2_updated, right1_updated, right2_updated]):
        logger.error("Failed to update some entities")
        return
        
    # Use updated versions
    left1 = cast(ManyToManyLeft, left1_updated)
    
    # Retrieve and check references
    retrieved_left1 = ManyToManyLeft.get(left1.ecs_id)
    if retrieved_left1:
        logger.info(f"Retrieved left1: {retrieved_left1.ecs_id}")
        logger.info(f"Left1 right_ids: {retrieved_left1.right_ids}")
        
        # Use property to get actual objects
        rights = retrieved_left1.rights
        logger.info(f"Left1 has {len(rights)} rights")
        for i, right in enumerate(rights):
            logger.info(f"  Right {i}: {right.name}, {right.ecs_id}")
            
            # Check back-references
            lefts = right.lefts
            logger.info(f"  Right {i} has {len(lefts)} lefts")
            for j, left in enumerate(lefts):
                logger.info(f"    Left {j}: {left.name}, {left.ecs_id}")

def test_hierarchical_relationship():
    """Test hierarchical relationships."""
    logger.info("=== TESTING HIERARCHICAL RELATIONSHIPS ===")
    
    # Create root first
    root = HierarchicalEntity(name="Root")
    root_result = EntityRegistry.register(root)
    if not root_result:
        logger.error("Failed to register root")
        return
    
    # Use registered version
    root = cast(HierarchicalEntity, root_result)
    logger.info(f"Created root: {root.ecs_id}")
    
    # Create children referencing root by ID
    child1 = HierarchicalEntity(name="Child1", parent_id=root.ecs_id)
    child2 = HierarchicalEntity(name="Child2", parent_id=root.ecs_id)
    
    child1_result = EntityRegistry.register(child1)
    child2_result = EntityRegistry.register(child2)
    
    if not all([child1_result, child2_result]):
        logger.error("Failed to register children")
        return
        
    # Use registered versions
    child1 = cast(HierarchicalEntity, child1_result)
    child2 = cast(HierarchicalEntity, child2_result)
    
    logger.info(f"Created child1: {child1.ecs_id}, child2: {child2.ecs_id}")
    
    # Create grandchildren
    grandchild1 = HierarchicalEntity(name="GrandChild1", parent_id=child1.ecs_id)
    grandchild2 = HierarchicalEntity(name="GrandChild2", parent_id=child1.ecs_id)
    
    grandchild1_result = EntityRegistry.register(grandchild1)
    grandchild2_result = EntityRegistry.register(grandchild2)
    
    if not all([grandchild1_result, grandchild2_result]):
        logger.error("Failed to register grandchildren")
        return
        
    # Use registered versions
    grandchild1 = cast(HierarchicalEntity, grandchild1_result)
    grandchild2 = cast(HierarchicalEntity, grandchild2_result)
    
    logger.info(f"Created grandchild1: {grandchild1.ecs_id}, grandchild2: {grandchild2.ecs_id}")
    
    # Update parent-child references
    root.child_ids = [child1.ecs_id, child2.ecs_id]
    child1.child_ids = [grandchild1.ecs_id, grandchild2.ecs_id]
    
    # Update in registry
    root_updated = EntityRegistry.register(root)
    child1_updated = EntityRegistry.register(child1)
    
    if not all([root_updated, child1_updated]):
        logger.error("Failed to update hierarchy")
        return
        
    # Use updated versions
    root = cast(HierarchicalEntity, root_updated)
    
    # Retrieve and navigate hierarchy
    retrieved_root = HierarchicalEntity.get(root.ecs_id)
    if retrieved_root:
        logger.info(f"Retrieved root: {retrieved_root.ecs_id}")
        logger.info(f"Root has {len(retrieved_root.child_ids)} children")
        
        # Navigate to children using property
        children = retrieved_root.children
        for i, child in enumerate(children):
            logger.info(f"  Child {i}: {child.name}, {child.ecs_id}")
            
            # Check grandchildren
            grandchildren = child.children
            logger.info(f"  Child {i} has {len(grandchildren)} children")
            for j, grandchild in enumerate(grandchildren):
                logger.info(f"    Grandchild {j}: {grandchild.name}, {grandchild.ecs_id}")
                
                # Check parent reference (upward)
                parent = grandchild.parent
                if parent:
                    logger.info(f"    Grandchild {j}'s parent: {parent.name}, {parent.ecs_id}")

def main():
    """Run all tests."""
    test_one_to_many_relationship()
    print("\n")
    test_many_to_many_relationship()
    print("\n")
    test_hierarchical_relationship()

if __name__ == "__main__":
    main()