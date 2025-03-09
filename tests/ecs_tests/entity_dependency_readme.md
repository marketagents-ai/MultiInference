# Entity Dependency Graph

This module provides utilities to build and analyze dependency graphs for Entity objects, with a focus on:

1. Detecting circular references without modifying the object model
2. Building a full dependency graph of entities
3. Providing a topological sort for bottom-up processing

## Key Concepts

### Dependency Graph

The dependency graph is a directed graph where:
- Nodes are entities (identified by their `ecs_id` or Python object ID)
- Edges are dependencies between entities (A depends on B if A has a reference to B)

### Cycle Detection

Cycles in the dependency graph represent circular references, where entity A depends on entity B, which depends on entity C, which depends back on entity A (or some variation).

For example:
```
root → child1 → grandchild → root  (Circular reference)
```

### Topological Sort

A topological sort orders entities such that if entity A depends on entity B, then B comes before A in the ordering. This is crucial for bottom-up processing in the entity forking process.

If cycles exist, a perfect topological sort is impossible, but we can still provide a best-effort ordering.

## Usage

```python
from entity_dependencies import EntityDependencyGraph, CycleStatus

# Create a graph
graph = EntityDependencyGraph()

# Build the graph starting from a root entity
status = graph.build_graph(root_entity)

# Check for cycles
if status == CycleStatus.CYCLE_DETECTED:
    print("Circular references detected:")
    for cycle in graph.get_cycles():
        print(f"  Cycle: {cycle}")
        
# Get a topological sort for processing
sorted_entities = graph.get_topological_sort()
```

## Integration with Entity System

This dependency graph can be integrated with the Entity system to:

1. **Detect Circular References**: Find circular dependencies in entity relationships without requiring ID-based references

2. **Optimize Forking**: Process entities in dependency order (bottom-up) for more efficient forking

3. **Visualize Relationships**: Generate visualizations of entity relationships for debugging

4. **Enable Safe Serialization**: Break dependency cycles when needed for serialization

## Implementation in Entity.py

To integrate this into `entity.py`, you would:

1. Create a method to compute the dependency graph on demand
2. Use the dependency graph to sort entities for processing in `fork()`
3. Add cycle checking for validation

Example integration:

```python
def get_dependency_graph(self) -> EntityDependencyGraph:
    """Get the dependency graph for this entity tree."""
    graph = EntityDependencyGraph()
    graph.build_graph(self)
    return graph
    
def get_sub_entities_topologically_sorted(self) -> List[Entity]:
    """Get sub-entities in topological order (dependencies first)."""
    graph = self.get_dependency_graph()
    return graph.get_topological_sort()
```

## Benefits of This Approach

The key benefit is that we maintain the natural object model where entities have direct references to each other, while still being able to detect and handle circular references when needed.

This is unlike the ID-based approach where we would have to replace object references with ID references, which would make the code less natural and more complex.