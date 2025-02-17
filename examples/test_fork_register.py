"""
Test script for verifying entity forking and registration patterns.
Tests various scenarios including:
1. Linear chain forking
2. Tree-like forking
3. Registration behavior
4. Cold snapshot management
5. Lineage tracking
"""
from pydantic import BaseModel, Field
from minference.entity import Entity, EntityRegistry, entity_uuid_expander, entity_uuid_expander_list
from typing import List, Dict, Any, Optional
from uuid import UUID
import logging
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Clear any existing state
registry = EntityRegistry()
EntityRegistry.clear()
EntityRegistry.clear_logs()

# Test entities
class SimpleEntity(Entity):
    """A simple entity with a few fields for testing."""
    value: str
    count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class NestedEntity(Entity):
    """Entity containing other entities to test nested forking."""
    name: str
    child: Optional[SimpleEntity] = None
    siblings: List[SimpleEntity] = Field(default_factory=list)

def print_entity_info(entity: Entity, prefix: str = "") -> None:
    """Helper function to print entity information."""
    print(f"{prefix}Entity ID: {entity.id}")
    print(f"{prefix}Lineage ID: {entity.lineage_id}")
    print(f"{prefix}Parent ID: {entity.parent_id}")
    print(f"{prefix}Data: {entity.entity_dump()}")
    print()

def test_linear_chain():
    """Test creating a linear chain of entity versions."""
    print("\n=== Test 1: Linear Chain Forking ===")
    
    # Create root entity
    root = SimpleEntity(value="initial")
    print("Created root entity:")
    print_entity_info(root)
    
    # Create first fork
    v1 = root.fork(value="modified1", count=1)
    print("Created first fork:")
    print_entity_info(v1)
    
    # Create second fork
    v2 = v1.fork(value="modified2", count=2)
    print("Created second fork:")
    print_entity_info(v2)
    
    # Verify lineage
    print("\nLineage verification:")
    print(EntityRegistry.get_lineage_mermaid(root.lineage_id))

def test_tree_forking():
    """Test creating a tree-like structure with multiple branches."""
    print("\n=== Test 2: Tree-like Forking ===")
    
    # Create root entity
    root = SimpleEntity(value="root")
    root_id = root.id  # Store the root ID before any modifications
    print("Created root entity:")
    print_entity_info(root)
    
    # Create branch 1 from root
    branch1 = root.fork(value="branch1")
    branch1_id = branch1.id  # Store branch1 ID
    print("\nCreated branch 1:")
    print_entity_info(branch1)
    
    # Get fresh copy of root from registry for branch 2
    root = EntityRegistry.get(root_id)
    if root is None:
        print("Error: Could not retrieve root entity")
        return
    
    # Create branch 2 from fresh copy of root
    branch2 = root.fork(value="branch2")
    branch2_id = branch2.id  # Store branch2 ID
    print("Created branch 2:")
    print_entity_info(branch2)
    
    # Create sub-branches
    leaf1 = branch1.fork(value="leaf1")
    leaf1_id = leaf1.id  # Store leaf1 ID
    print("\nCreated leaf 1 (from branch 1):")
    print_entity_info(leaf1)
    
    leaf2 = branch2.fork(value="leaf2")
    leaf2_id = leaf2.id  # Store leaf2 ID
    print("Created leaf 2 (from branch 2):")
    print_entity_info(leaf2)
    
    # Verify tree structure
    print("\nTree structure verification:")
    print(EntityRegistry.get_lineage_mermaid(root.lineage_id))
    
    # Additional verification
    print("\nVerifying branch relationships:")
    print(f"Root ID: {root_id}")
    
    # Get fresh copies from registry for verification
    branch1_fresh = EntityRegistry.get(branch1_id)  # Use stored ID
    branch2_fresh = EntityRegistry.get(branch2_id)  # Use stored ID
    leaf1_fresh = EntityRegistry.get(leaf1_id)    # Use stored ID
    leaf2_fresh = EntityRegistry.get(leaf2_id)    # Use stored ID
    
    if not all([branch1_fresh, branch2_fresh, leaf1_fresh, leaf2_fresh]):
        print("Error: Failed to retrieve one or more entities from registry")
        return
    
    # Type assertions for linter
    assert isinstance(branch1_fresh, SimpleEntity)
    assert isinstance(branch2_fresh, SimpleEntity)
    assert isinstance(leaf1_fresh, SimpleEntity)
    assert isinstance(leaf2_fresh, SimpleEntity)
    
    print(f"Branch 1: ID={branch1_fresh.id}, Parent ID={branch1_fresh.parent_id}")
    print(f"Branch 2: ID={branch2_fresh.id}, Parent ID={branch2_fresh.parent_id}")
    print(f"Leaf 1: ID={leaf1_fresh.id}, Parent ID={leaf1_fresh.parent_id}")
    print(f"Leaf 2: ID={leaf2_fresh.id}, Parent ID={leaf2_fresh.parent_id}")
    print(f"\nVerifying relationships are correct:")
    print(f"Branch 1 parent is root: {branch1_fresh.parent_id == root_id}")  # Should be True
    print(f"Branch 2 parent is root: {branch2_fresh.parent_id == root_id}")  # Should be True
    print(f"Leaf 1 parent is branch1: {leaf1_fresh.parent_id == branch1_id}")  # Should be True
    print(f"Leaf 2 parent is branch2: {leaf2_fresh.parent_id == branch2_id}")  # Should be True
    print(f"All share same lineage: {len({root.lineage_id, branch1_fresh.lineage_id, branch2_fresh.lineage_id, leaf1_fresh.lineage_id, leaf2_fresh.lineage_id}) == 1}")

def test_nested_entities():
    """Test forking behavior with nested entities."""
    print("\n=== Test 3: Nested Entity Forking ===")
    
    # Create nested structure
    child = SimpleEntity(value="child")
    sibling1 = SimpleEntity(value="sibling1")
    sibling2 = SimpleEntity(value="sibling2")
    
    root = NestedEntity(
        name="root",
        child=child,
        siblings=[sibling1, sibling2]
    )
    print("Created nested structure:")
    print_entity_info(root)
    
    # Modify nested entity
    modified_child = SimpleEntity(value="modified_child")
    v1 = root.fork(child=modified_child)
    print("\nModified child entity:")
    print_entity_info(v1)
    
    # Modify sibling list
    v2 = v1.fork(siblings=[SimpleEntity(value="new_sibling")])
    print("\nModified siblings:")
    print_entity_info(v2)
    
    # Verify nested structure
    print("\nNested structure verification:")
    print(EntityRegistry.get_lineage_mermaid(root.lineage_id))

def test_cold_snapshots():
    """Test cold snapshot behavior and registration."""
    print("\n=== Test 4: Cold Snapshot Management ===")
    
    # Create and modify entity without forking
    entity = SimpleEntity(value="original")
    print("Original entity:")
    print_entity_info(entity)
    
    # Get cold snapshot
    cold_snapshot = EntityRegistry.get_cold_snapshot(entity.id)
    if cold_snapshot is None:
        print("\nNo cold snapshot found!")
        return
    print("\nCold snapshot:")
    print_entity_info(cold_snapshot)
    
    # Modify without forking
    entity.value = "modified"
    print("\nModified entity (before fork):")
    print_entity_info(entity)
    
    # Compare with cold snapshot
    print("\nComparison with cold snapshot:")
    print(f"Has modifications: {entity.has_modifications(cold_snapshot)}")
    print(f"Differences: {entity.compute_diff(cold_snapshot).field_diffs}")
    
    # Force fork
    forked = entity.fork(force=True)
    print("\nForced fork:")
    print_entity_info(forked)

def test_registration_behavior():
    """Test entity registration behavior."""
    print("\n=== Test 5: Registration Behavior ===")
    
    # Create entity and check registration
    entity = SimpleEntity(value="test")
    print("Initial registration:")
    print(f"Entity registered: {EntityRegistry.has_entity(entity.id)}")
    print_entity_info(entity)
    
    # Modify and check auto-registration
    modified = entity.fork(value="modified")
    print("\nAfter modification:")
    print(f"Original still registered: {EntityRegistry.has_entity(entity.id)}")
    print(f"Modified version registered: {EntityRegistry.has_entity(modified.id)}")
    print_entity_info(modified)
    
    # Check registry status
    status = EntityRegistry.get_registry_status()
    print("\nRegistry status:")
    print(f"Total entities: {status['total_items']}")
    print(f"Total lineages: {status['total_lineages']}")
    print(f"Total versions: {status['total_versions']}")

@entity_uuid_expander("entity")
async def modify_entity_tree(entity: SimpleEntity, new_value: str) -> SimpleEntity:
    """Test function using the decorator to modify an entity in the tree."""
    return entity.fork(value=new_value)

@entity_uuid_expander_list("entities")
def modify_entities_sync(entities: List[SimpleEntity], new_value: str) -> List[SimpleEntity]:
    """Test function using the list decorator to modify multiple entities synchronously."""
    return [entity.fork(value=f"{new_value}_{i}") for i, entity in enumerate(entities)]

@entity_uuid_expander_list("entities")
async def modify_entities_async(entities: List[SimpleEntity], new_value: str) -> List[SimpleEntity]:
    """Test function using the list decorator to modify multiple entities asynchronously."""
    return [entity.fork(value=f"{new_value}_{i}") for i, entity in enumerate(entities)]

def test_decorator_tree():
    """Test using entity_uuid_expander with tree structure."""
    print("\n=== Test 6: Decorator with Tree Structure ===")
    
    # Create root entity
    root = SimpleEntity(value="root")
    root_id = root.id
    print("Created root entity:")
    print_entity_info(root)
    
    # Create branches
    branch1 = root.fork(value="branch1")
    branch1_id = branch1.id
    print("\nCreated branch 1:")
    print_entity_info(branch1)
    
    # Get fresh copy of root
    root = EntityRegistry.get(root_id)
    if root is None:
        print("Error: Could not retrieve root entity")
        return
        
    branch2 = root.fork(value="branch2")
    branch2_id = branch2.id
    print("Created branch 2:")
    print_entity_info(branch2)
    
    # Use decorator to modify branches
    print("\nModifying branches using single entity decorator:")
    asyncio.run(modify_entity_tree(branch1, "modified_branch1"))
    asyncio.run(modify_entity_tree(branch2, "modified_branch2"))
    
    # Verify modifications
    print("\nVerifying modifications:")
    branch1_fresh = EntityRegistry.get(branch1_id)
    branch2_fresh = EntityRegistry.get(branch2_id)
    
    if not all([branch1_fresh, branch2_fresh]):
        print("Error: Failed to retrieve entities from registry")
        return
        
    assert isinstance(branch1_fresh, SimpleEntity)
    assert isinstance(branch2_fresh, SimpleEntity)
    
    print(f"Branch 1 value: {branch1_fresh.value}")
    print(f"Branch 2 value: {branch2_fresh.value}")
    
    # Verify tree structure
    print("\nTree structure verification:")
    print(EntityRegistry.get_lineage_mermaid(root.lineage_id))

def test_list_decorators():
    """Test using entity_uuid_expander_list with both sync and async functions."""
    print("\n=== Test 7: List Decorators ===")
    
    # Create a set of entities
    entities = [
        SimpleEntity(value=f"entity_{i}") 
        for i in range(3)
    ]
    entity_ids = [e.id for e in entities]
    
    print("Created initial entities:")
    for entity in entities:
        print_entity_info(entity)
    
    # Test synchronous modification
    print("\nModifying entities using synchronous list decorator:")
    modified_sync = modify_entities_sync(entities, "sync_modified")
    modified_sync_ids = [e.id for e in modified_sync]
    
    print("\nVerifying synchronous modifications:")
    print("Original entities after sync modification:")
    for i, entity_id in enumerate(entity_ids):
        fresh_entity = EntityRegistry.get(entity_id)
        if fresh_entity:
            print(f"Original Entity {i} value: {fresh_entity.value}")
    
    print("\nNew entities from sync modification:")
    for i, entity_id in enumerate(modified_sync_ids):
        fresh_entity = EntityRegistry.get(entity_id)
        if fresh_entity:
            print(f"Modified Entity {i} value: {fresh_entity.value}")
    
    # Test asynchronous modification
    print("\nModifying entities using asynchronous list decorator:")
    modified_async = asyncio.run(modify_entities_async(modified_sync, "async_modified"))
    modified_async_ids = [e.id for e in modified_async]
    
    print("\nVerifying asynchronous modifications:")
    print("Sync modified entities after async modification:")
    for i, entity_id in enumerate(modified_sync_ids):
        fresh_entity = EntityRegistry.get(entity_id)
        if fresh_entity:
            print(f"Sync Entity {i} value: {fresh_entity.value}")
    
    print("\nNew entities from async modification:")
    for i, entity_id in enumerate(modified_async_ids):
        fresh_entity = EntityRegistry.get(entity_id)
        if fresh_entity:
            print(f"Async Entity {i} value: {fresh_entity.value}")
    
    # Verify lineage for one of the entities
    print("\nLineage verification for first entity:")
    print(EntityRegistry.get_lineage_mermaid(entities[0].lineage_id))
    
    # Print all versions in order for the first entity
    print("\nAll versions for first entity in order:")
    first_entity_lineage = EntityRegistry.get_lineage_tree_sorted(entities[0].lineage_id)
    if first_entity_lineage and first_entity_lineage["nodes"]:
        for node_id in first_entity_lineage["sorted_ids"]:
            node = first_entity_lineage["nodes"][node_id]
            entity = node["entity"]
            print(f"ID: {entity.id}, Value: {entity.value}, Parent: {entity.parent_id}")

def main():
    """Run all tests."""
    # Run individual tests
    test_linear_chain()
    test_tree_forking()
    test_nested_entities()
    test_cold_snapshots()
    test_registration_behavior()
    test_decorator_tree()
    test_list_decorators()
    
    # Print final registry status
    print("\n=== Final Registry Status ===")
    status = EntityRegistry.get_registry_status()
    print(f"Total entities: {status['total_items']}")
    print(f"Entities by type: {status['entities_by_type']}")
    print(f"Total lineages: {status['total_lineages']}")
    print(f"Total versions: {status['total_versions']}")
    print(f"Version history: {status['version_history']}")

if __name__ == "__main__":
    main() 