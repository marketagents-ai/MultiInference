"""
Implementation of entity dependency graph algorithm.

This module provides utilities to compute the dependency graph of entities
and detect circular references without modifying the underlying object model.
"""
import sys
from uuid import UUID, uuid4
from typing import Dict, Set, List, Optional, Any, Tuple, TypeVar, Callable
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("entity_dependencies")

# For demo/testing purposes to avoid import issues
class Entity:
    """Mock Entity class for demo purposes."""
    def __init__(self, id=None):
        self.ecs_id = id or uuid4()
        
    def __str__(self):
        return f"Entity({self.ecs_id})"
        
    def __repr__(self):
        return self.__str__()

class GraphNode:
    """Represents a node in the entity dependency graph."""
    def __init__(self, entity: Any):
        self.entity = entity
        self.entity_id = getattr(entity, "ecs_id", id(entity))
        self.dependencies = set()  # IDs of entities this entity depends on
        self.dependents = set()    # IDs of entities that depend on this entity
        
    def add_dependency(self, dep_id: Any):
        """Add a dependency to this node."""
        self.dependencies.add(dep_id)
        
    def add_dependent(self, dep_id: Any):
        """Add a dependent to this node."""
        self.dependents.add(dep_id)
        
    def __str__(self):
        return f"Node({self.entity_id}, deps={len(self.dependencies)}, dependents={len(self.dependents)})"
        
    def __repr__(self):
        return self.__str__()

class CycleStatus(Enum):
    """Status of cycle detection."""
    NO_CYCLE = 0
    CYCLE_DETECTED = 1
    
class EntityDependencyGraph:
    """
    Computes and maintains the dependency graph of entities.
    
    This class provides methods to:
    1. Build the dependency graph of a root entity
    2. Detect cycles in the graph
    3. Get topological sort of entities (for bottom-up processing)
    4. Add/remove entities to the graph
    5. Query entity relationships in the graph
    """
    def __init__(self):
        self.nodes: Dict[Any, GraphNode] = {}  # Map of entity ID to its node
        self.cycles: List[List[Any]] = []      # List of detected cycles
        
    def build_graph(self, root_entity: Any, 
                    is_entity_check=None,
                    get_entity_id=None) -> CycleStatus:
        """
        Build the dependency graph starting from a root entity.
        
        Args:
            root_entity: The root entity to start from
            is_entity_check: Optional function to determine if an object is an entity
            get_entity_id: Optional function to get an entity's ID
            
        Returns:
            CycleStatus indicating if any cycles were detected
        """
        logger.debug(f"Building dependency graph for {root_entity}")
        
        # Default entity detection function
        if is_entity_check is None:
            def is_entity_func(obj):
                return hasattr(obj, "ecs_id")
        else:
            is_entity_func = is_entity_check
                
        # Default ID function
        if get_entity_id is None:
            def get_entity_id_func(entity):
                return getattr(entity, "ecs_id", id(entity))
        else:
            get_entity_id_func = get_entity_id
                
        # Clear existing graph
        self.nodes.clear()
        self.cycles.clear()
        
        # First, collect all entities and their dependencies
        entities_to_process = [root_entity]
        entity_dependencies = {}  # Map entity ID to set of dependency IDs
        
        # Collect all entities and their immediate dependencies
        while entities_to_process:
            entity = entities_to_process.pop(0)
            entity_id = get_entity_id_func(entity)
            
            # Skip if already processed
            if entity_id in entity_dependencies:
                continue
                
            # Add to graph
            if entity_id not in self.nodes:
                self.nodes[entity_id] = GraphNode(entity)
            
            # Find immediate dependencies
            deps = self._find_entity_references(entity, is_entity_func, get_entity_id_func)
            dep_ids = set()
            
            for dep_entity, _ in deps:
                dep_id = get_entity_id_func(dep_entity)
                dep_ids.add(dep_id)
                
                # Create node if needed
                if dep_id not in self.nodes:
                    self.nodes[dep_id] = GraphNode(dep_entity)
                
                # Set up bidirectional relationship
                self.nodes[entity_id].add_dependency(dep_id)
                self.nodes[dep_id].add_dependent(entity_id)
                
                # Add to processing queue if not already processed
                if dep_id not in entity_dependencies:
                    entities_to_process.append(dep_entity)
            
            # Store dependencies
            entity_dependencies[entity_id] = dep_ids
            
        # Now detect cycles using cycle finding algorithm (DFS)
        has_cycle = False
        visited = set()  # Nodes we've fully processed
        path = set()     # Nodes in current path
        
        def find_cycles(node_id):
            nonlocal has_cycle
            
            # If already visited, no need to process again
            if node_id in visited:
                return
                
            # If in current path, we found a cycle
            if node_id in path:
                # Found a cycle, reconstruct it
                cycle = []
                for n in list(path) + [node_id]:
                    cycle.append(n)
                    if n == node_id and len(cycle) > 1:
                        break
                
                logger.warning(f"Detected cycle: {cycle}")
                self.cycles.append(cycle)
                has_cycle = True
                return
                
            # Add to current path
            path.add(node_id)
            
            # Process all dependencies
            if node_id in entity_dependencies:
                for dep_id in entity_dependencies[node_id]:
                    find_cycles(dep_id)
                    
            # Remove from path and mark as visited
            path.remove(node_id)
            visited.add(node_id)
            
        # Look for cycles starting from each node
        for node_id in self.nodes:
            find_cycles(node_id)
            
        logger.info(f"Built dependency graph with {len(self.nodes)} nodes")
        if self.cycles:
            logger.warning(f"Detected {len(self.cycles)} cycles in the graph")
            has_cycle = True
            
        return CycleStatus.CYCLE_DETECTED if has_cycle else CycleStatus.NO_CYCLE
        
    def add_entity(self, entity: Any, dependencies: Optional[List[Any]] = None) -> None:
        """
        Add an entity to the graph with optional dependencies.
        
        Args:
            entity: The entity to add
            dependencies: Optional list of entities this entity depends on
        """
        # Get entity ID
        entity_id = getattr(entity, "ecs_id", id(entity))
        
        # Create node if needed
        if entity_id not in self.nodes:
            self.nodes[entity_id] = GraphNode(entity)
            
        # Add dependencies if provided
        if dependencies:
            for dep in dependencies:
                if dep is not None:
                    dep_id = getattr(dep, "ecs_id", id(dep))
                    
                    # Create node for dependency if needed
                    if dep_id not in self.nodes:
                        self.nodes[dep_id] = GraphNode(dep)
                        
                    # Set up bidirectional relationship
                    self.nodes[entity_id].add_dependency(dep_id)
                    self.nodes[dep_id].add_dependent(entity_id)
    
    def get_node(self, entity_id: Any) -> Optional[GraphNode]:
        """Get a node by entity ID."""
        return self.nodes.get(entity_id)
        
    def get_dependent_ids(self, entity_id: Any) -> Set[Any]:
        """
        Get IDs of entities that depend on this entity.
        
        Args:
            entity_id: ID of the entity to get dependents for
            
        Returns:
            Set of dependent entity IDs
        """
        node = self.get_node(entity_id)
        if node:
            return node.dependents
        return set()
        
    def is_graph_root(self, entity_id: Any) -> bool:
        """
        Check if entity is a root entity in the graph.
        
        A root entity is one that has no parent dependencies 
        (i.e., no other entity depends on it).
        
        Args:
            entity_id: ID of the entity to check
            
        Returns:
            True if entity is a root entity
        """
        node = self.get_node(entity_id)
        if node:
            return len(node.dependents) == 0
        return True  # If not in graph, consider it a root
    
    def _find_entity_references(self, obj: Any, 
                               is_entity_func, 
                               get_entity_id_func) -> List[Tuple[Any, str]]:
        """
        Find all entity references in an object.
        
        Args:
            obj: Object to inspect
            is_entity_func: Function to determine if an object is an entity
            get_entity_id_func: Function to get an entity's ID
            
        Returns:
            List of (entity, path) tuples
        """
        results = []
        
        # Skip None values
        if obj is None:
            return results
        
        # Debug
        logger.debug(f"Finding entity references in {obj}")
            
        # Process different container types
        if isinstance(obj, dict):
            for key, value in obj.items():
                if is_entity_func(value):
                    logger.debug(f"Found entity in dict at key {key}: {value}")
                    results.append((value, f"{key}"))
                elif isinstance(value, (dict, list, tuple, set)):
                    # Recursively process containers
                    nested = self._find_entity_references(
                        value, is_entity_func, get_entity_id_func)
                    results.extend([(e, f"{key}.{p}") for e, p in nested])
                    
        elif isinstance(obj, (list, tuple, set)):
            for i, value in enumerate(obj):
                if is_entity_func(value):
                    logger.debug(f"Found entity in list at index {i}: {value}")
                    results.append((value, f"[{i}]"))
                elif isinstance(value, (dict, list, tuple, set)):
                    # Recursively process containers
                    nested = self._find_entity_references(
                        value, is_entity_func, get_entity_id_func)
                    results.extend([(e, f"[{i}].{p}") for e, p in nested])
                    
        else:
            # Get list of object attributes, filtering non-property attributes
            attr_names = []
            for attr_name in dir(obj):
                # Skip private attributes and methods
                if attr_name.startswith('_'):
                    continue
                    
                try:
                    attr = getattr(type(obj), attr_name, None)
                    if attr is not None and isinstance(attr, property):
                        # This is a property - include it
                        attr_names.append(attr_name)
                    elif not callable(getattr(obj, attr_name)):
                        # This is a regular attribute, not a method - include it
                        attr_names.append(attr_name)
                except:
                    pass
                        
            logger.debug(f"Inspecting attributes of {obj}: {attr_names}")
            
            # For each attribute, check if it's an entity
            for attr_name in attr_names:                    
                try:
                    value = getattr(obj, attr_name)
                    
                    # Check if attribute is an entity
                    if value is not None and is_entity_func(value):
                        logger.debug(f"Found entity in attribute {attr_name}: {value}")
                        results.append((value, attr_name))
                    elif isinstance(value, (dict, list, tuple, set)):
                        # Recursively process containers
                        nested = self._find_entity_references(
                            value, is_entity_func, get_entity_id_func)
                        results.extend([(e, f"{attr_name}.{p}") for e, p in nested])
                except (AttributeError, TypeError) as e:
                    # Skip attributes that can't be accessed
                    logger.debug(f"Error accessing attribute {attr_name}: {e}")
                    pass
                    
        logger.debug(f"Found {len(results)} entity references in {obj}")
        return results
    
    def get_topological_sort(self) -> List[Any]:
        """
        Return entities in topological order (dependencies first).
        
        This ensures that when processing entities, all dependencies
        are processed before their dependents.
        
        Returns:
            List of entity IDs in topological order
        """
        # Find all nodes without dependencies
        result = []
        visited = set()
        
        # For each node, calculate its depth in the dependency graph
        depths = {}
        
        def calculate_depth(node_id, path=None):
            """Calculate maximum dependency depth for a node."""
            if path is None:
                path = set()
                
            # Check for cycles - if we've seen this node before in this path
            if node_id in path:
                return 0
                
            # If already calculated, return cached value
            if node_id in depths:
                return depths[node_id]
                
            path = path.union({node_id})
            
            # If no dependencies, depth is 0
            node = self.nodes[node_id]
            if not node.dependencies:
                depths[node_id] = 0
                return 0
                
            # Calculate maximum depth of dependencies
            max_depth = 0
            for dep_id in node.dependencies:
                if dep_id in self.nodes:
                    depth = calculate_depth(dep_id, path) + 1
                    max_depth = max(max_depth, depth)
                    
            depths[node_id] = max_depth
            return max_depth
            
        # Calculate depth for all nodes
        for node_id in self.nodes:
            if node_id not in depths:
                calculate_depth(node_id)
                
        # Sort nodes by depth (lowest first)
        sorted_nodes = sorted(self.nodes.keys(), key=lambda node_id: depths.get(node_id, 0))
        
        # Return the actual entities
        return [self.nodes[node_id].entity for node_id in sorted_nodes]
    
    def get_cycles(self) -> List[List[Any]]:
        """Get all detected cycles in the graph."""
        return self.cycles
        
    def find_entity_by_id(self, entity_id: Any) -> Optional[Any]:
        """Find an entity by its ID."""
        if entity_id in self.nodes:
            return self.nodes[entity_id].entity
        return None

def test_dependency_graph():
    """Test the dependency graph with simple entities."""
    # Create some test entities
    root = Entity()
    child1 = Entity()
    child2 = Entity()
    grandchild = Entity()
    
    print(f"Root: {root.ecs_id}")
    print(f"Child1: {child1.ecs_id}")
    print(f"Child2: {child2.ecs_id}")
    print(f"Grandchild: {grandchild.ecs_id}")
    
    # Set up relationships (mock implementation for testing)
    setattr(root, "children", [child1, child2])
    setattr(child1, "parent", root)
    setattr(child1, "child", grandchild)
    setattr(child2, "parent", root)
    setattr(grandchild, "parent", child1)
    
    # Create circular reference
    setattr(grandchild, "root", root)
    
    # Print entity attributes
    print("\nEntity attributes:")
    for entity, name in [(root, "root"), (child1, "child1"), (child2, "child2"), (grandchild, "grandchild")]:
        print(f"{name} attributes:")
        for attr in dir(entity):
            if not attr.startswith("_") and not callable(getattr(entity, attr)):
                value = getattr(entity, attr)
                if isinstance(value, Entity):
                    print(f"  {attr}: Entity({value.ecs_id})")
                elif isinstance(value, list) and value and isinstance(value[0], Entity):
                    print(f"  {attr}: [{', '.join(f'Entity({e.ecs_id})' for e in value)}]")
                else:
                    print(f"  {attr}: {value}")
    
    # Build graph with verbose logging
    logging.getLogger("entity_dependencies").setLevel(logging.DEBUG)
    graph = EntityDependencyGraph()
    print("\nBuilding dependency graph...")
    status = graph.build_graph(root)
    
    # Check result
    print(f"\nGraph status: {status}")
    print(f"Number of nodes: {len(graph.nodes)}")
    print(f"Cycles: {graph.get_cycles()}")
    
    # Print node info
    print("\nGraph nodes:")
    for node_id, node in graph.nodes.items():
        print(f"Node {node_id}:")
        print(f"  Dependencies: {node.dependencies}")
        print(f"  Dependents: {node.dependents}")
    
    # Get topological sort
    sorted_entities = graph.get_topological_sort()
    print("\nTopological sort:")
    for entity in sorted_entities:
        print(f"  {entity}")

if __name__ == "__main__":
    test_dependency_graph()