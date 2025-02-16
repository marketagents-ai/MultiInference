from pydantic import BaseModel, Field
from minference.entity import Entity, EntityRegistry, entity_uuid_expander, entity_uuid_expander_list
from typing import List

registry = EntityRegistry()
EntityRegistry.clear()
EntityRegistry.clear_logs()

class MyEntity(Entity):
    some_data: str = "initial"

# Example usage of snapshot-based re-registration
root = MyEntity(some_data="Hello")
EntityRegistry.register(root)
root.some_data = "Changed"
EntityRegistry.register(root)
lineage = EntityRegistry.get_lineage_ids(root.lineage_id)
print("Lineage =>", lineage)

@entity_uuid_expander
def single_func(e: MyEntity) -> None:
    print("single_func => got entity", e.id, e.some_data)
    e.some_data = "modified in single_func"

# Call single_func with either entity or UUID
single_func(root)
single_func(lineage[0])

@entity_uuid_expander_list("items")
async def run_parallel_ai_completion(items: List[MyEntity], orchestrator: str) -> List[str]:
    # items is guaranteed to be List[MyEntity] at runtime
    out = []
    for x in items:
        out.append(f"Processed {x.id} - {x.some_data}")
        x.some_data = f"modified by {orchestrator}"
    return out

# Create some test entities
entities = [MyEntity(some_data=f"test_{i}") for i in range(3)]
entity_ids = [e.id for e in entities]

# Demo both ways of calling the decorated function
import asyncio

async def run_demo():
    # Call with List[Entity]
    results1 = await run_parallel_ai_completion(entities, "demo1")
    print("\nResults with entities:", results1)
    
    # Call with List[UUID]
    results2 = await run_parallel_ai_completion(entity_ids, "demo2")
    print("\nResults with UUIDs:", results2)
    
    # Show final registry state
    print("\nFinal registry status:")
    print(EntityRegistry.get_registry_status())

asyncio.run(run_demo())