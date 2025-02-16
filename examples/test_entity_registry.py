from pydantic import BaseModel, Field
from minference.entity import Entity, EntityRegistry, entity_uuid_expander, entity_uuid_expander_list
from typing import List, Union
from uuid import UUID

registry = EntityRegistry()
EntityRegistry.clear()
EntityRegistry.clear_logs()

class MyEntity(Entity):
    some_data: str = "initial"

# Create and register root entity
root = MyEntity(some_data="Hello")

# Modify and register changes
root.some_data = "Changed"

@entity_uuid_expander
def single_func(e: MyEntity) -> None:
    print("single_func => got entity", e.id, e.some_data)
    e.some_data = "modified in single_func"
      # This should be depth 2

# Call single_func with the root entity
single_func(root)

@entity_uuid_expander_list("items")
async def run_parallel_ai_completion(orchestrator: str, items: List[MyEntity]) -> List[str]:
    out = []
    for x in items:  # items will already be MyEntity instances
        out.append(f"Processed {x.id} - {x.some_data}")
        x.some_data = f"modified by {orchestrator}"
    return out

# Create test entities
entities = [MyEntity(some_data=f"test_{i}") for i in range(3)]


# Demo both ways of calling the decorated function
import asyncio

async def run_demo():
    # Call with List[Entity]
    results1 = await run_parallel_ai_completion("demo1", items=entities)
    print("\nResults with entities:", results1)
    
    # Call with List[UUID]
    entity_ids = [e.id for e in entities]
    results2 = await run_parallel_ai_completion("demo2", items=entity_ids)
    print("\nResults with UUIDs:", results2)
    
    # Show final registry state
    print("\nFinal registry status:")
    print(EntityRegistry.get_registry_status())
    
    # Print mermaid diagrams for both root and first test entity
    print("\nRoot entity lineage (Mermaid):")
    print(EntityRegistry.get_lineage_tree_sorted(root.lineage_id))
    
    print("\nTest entity lineage (Mermaid):")
    print(EntityRegistry.get_lineage_tree_sorted(entities[0].lineage_id))

asyncio.run(run_demo())