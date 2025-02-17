from pydantic import BaseModel, Field
from minference.entity import Entity, EntityRegistry, entity_uuid_expander, entity_uuid_expander_list
from typing import List, Union, Dict, Any, Optional
from uuid import UUID
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Clear any existing state
registry = EntityRegistry()
EntityRegistry.clear()
EntityRegistry.clear_logs()

# Test entities with various relationships
class NestedEntity(Entity):
    """A simple entity that can be nested in others"""
    value: str
    metadata: Dict[str, str] = Field(default_factory=dict)

class ComplexEntity(Entity):
    """An entity containing lists and nested entities"""
    name: str
    tags: List[str] = Field(default_factory=list)
    nested: Optional[NestedEntity] = None
    related: List[NestedEntity] = Field(default_factory=list)

def print_entity_info(entity: Entity, prefix: str = "") -> None:
    """Helper function to print entity information"""
    print(f"{prefix}Entity ID: {entity.id}")
    print(f"{prefix}Lineage ID: {entity.lineage_id}")
    print(f"{prefix}Parent ID: {entity.parent_id}")
    print(f"{prefix}Data: {entity.model_dump(exclude={'id', 'created_at', 'lineage_id', 'parent_id'})}")
    print()

# Test Scenario 1: Simple nested entity changes
print("=== Test Scenario 1: Nested Entity Changes ===")
print("\nCreating root entity with nested data")
nested1 = NestedEntity(value="v1", metadata={"type": "test"})
root = ComplexEntity(name="root", nested=nested1)
print_entity_info(root)

print("\nModifying nested entity")
nested2 = NestedEntity(value="v2", metadata={"type": "test", "status": "updated"})
v2 = root.fork(nested=nested2)
print_entity_info(v2)

# Test Scenario 2: List modifications
print("\n=== Test Scenario 2: List Modifications ===")
print("\nAdding items to lists")
v3 = v2.fork(
    tags=["tag1", "tag2"],
    related=[
        NestedEntity(value="related1"),
        NestedEntity(value="related2")
    ]
)
print_entity_info(v3)

print("\nModifying and removing list items")
v4 = v3.fork(
    tags=["tag1", "tag3"],  # Changed one tag
    related=[
        NestedEntity(value="related1"),  # Kept one, removed one
        NestedEntity(value="related3")   # Added new one
    ]
)
print_entity_info(v4)

print("\n=== Final Registry Status ===")
status = EntityRegistry.get_registry_status()
print(f"Total entities: {status['total_items']}")
print(f"Total lineages: {status['total_lineages']}")
print(f"Total versions: {status['total_versions']}")

print("\n=== Lineage Tree with Diffs ===")
tree = EntityRegistry.get_lineage_tree_sorted(root.lineage_id)

def print_node(node_id: UUID, tree_data: Dict[str, Any], indent: str = "") -> None:
    node = tree_data["nodes"][node_id]
    entity = node["entity"]
    print(f"{indent}Node: {node_id}")
    print(f"{indent}Type: {type(entity).__name__}")
    print(f"{indent}Data: {entity.model_dump(exclude={'id', 'created_at', 'lineage_id', 'parent_id'})}")
    
    # Print diffs if this is not the root
    if node["parent_id"]:
        print(f"{indent}Changes from parent:")
        if node_id in tree_data["diffs"]:
            for field, (action, value) in tree_data["diffs"][node_id].items():
                if action == "modified":
                    if isinstance(value, dict):
                        print(f"{indent}  {field}: {value['old_value']} -> {value['new_value']}")
                    else:
                        print(f"{indent}  {field}: {value}")
                else:
                    print(f"{indent}  {field}: {action} ({value})")
        else:
            print(f"{indent}  No changes")
    print()
    
    # Print children
    for child_id in sorted(node["children"]):  # Sort children for consistent output
        print_node(child_id, tree_data, indent + "  ")

# Print the tree starting from root
if tree["root"]:
    print_node(tree["root"], tree)

# Print raw diffs for debugging
print("=== Raw Diffs ===")
for node_id, diff in tree["diffs"].items():
    print(f"Node {node_id}:")
    print(f"  {diff}")
print()

# Print Mermaid visualization
print("\n=== Mermaid Visualization ===")
print(EntityRegistry.get_lineage_mermaid(root.lineage_id))

# Exit to prevent running old test code
import sys
sys.exit(0)