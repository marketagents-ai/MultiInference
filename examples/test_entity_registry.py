from pydantic import BaseModel, Field
from minference.entity import Entity, EntityRegistry, entity_uuid_expander, entity_uuid_expander_list
from typing import List, Union
from uuid import UUID
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Clear any existing state
registry = EntityRegistry()
EntityRegistry.clear()
EntityRegistry.clear_logs()

class MyEntity(Entity):
    some_data: str = "initial"

def print_entity_info(entity: MyEntity, prefix: str = "") -> None:
    """Helper function to print entity information"""
    print(f"{prefix}Entity ID: {entity.id}")
    print(f"{prefix}Lineage ID: {entity.lineage_id}")
    print(f"{prefix}Parent ID: {entity.parent_id}")
    print(f"{prefix}Data: {entity.some_data}")
    print()

print("=== Creating root entity (v1) ===")
root = MyEntity(some_data="v1")
print_entity_info(root)

print("=== Creating second version (v2) ===")
v2 = root.fork(some_data="v2")
print_entity_info(v2)

print("=== Creating third version (v3) ===")
v3 = v2.fork(some_data="v3")
print_entity_info(v3)

print("\n=== Final Registry Status ===")
status = EntityRegistry.get_registry_status()
print(f"Total entities: {status['total_items']}")
print(f"Total lineages: {status['total_lineages']}")
print(f"Total versions: {status['total_versions']}")

print("\n=== Lineage Tree ===")
print(EntityRegistry.get_lineage_tree_sorted(root.lineage_id))

# Exit immediately to prevent running old test code
import sys
sys.exit(0)