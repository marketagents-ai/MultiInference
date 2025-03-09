"""
Implementation of entity dependency graph algorithm.

This module provides utilities to compute the dependency graph of entities
and detect circular references without modifying the underlying object model.
"""
import sys
from uuid import UUID, uuid4
from typing import Dict, Set, List, Optional, Any, Tuple, TypeVar, Callable, cast
from enum import Enum
import logging
from pydantic import BaseModel, Field, ConfigDict

# Configure logging
logger = logging.getLogger("entity_dependencies")

class CycleStatus(Enum):
    """Status of cycle detection."""
    NO_CYCLE = 0
    CYCLE_DETECTED = 1

class GraphNode(BaseModel):
    """Represents a node in the entity dependency graph."""
    # Using Any type for entity since we can't import Entity here without circular imports
    entity: Any = Field(exclude=True)  # The entity object (excluded from serialization)
    entity_id: Any  # UUID or other identifier for the entity
    dependencies: Set[Any] = Field(default_factory=set)  # IDs of entities this entity depends on
    dependents: Set[Any] = Field(default_factory=set)  # IDs of entities that depend on this entity

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def add_dependency(self, dep_id: Any) -> None:
        """Add a dependency to this node."""
        self.dependencies.add(dep_id)
        
    def add_dependent(self, dep_id: Any) -> None:
        """Add a dependent to this node."""
        self.dependents.add(dep_id)
        
    def __str__(self) -> str:
        return f"Node({self.entity_id}, deps={len(self.dependencies)}, dependents={len(self.dependents)})"
        
    def __repr__(self) -> str:
        return self.__str__()

class EntityDependencyGraph(BaseModel):
    """
    Computes and maintains the dependency graph of entities.
    
    This class provides methods to:
    1. Build the dependency graph of a root entity
    2. Detect cycles in the graph
    3. Get topological sort of entities (for bottom-up processing)
    4. Add/remove entities to the graph
    5. Query entity relationships in the graph
    """
    nodes: Dict[Any, GraphNode] = Field(default_factory=dict)  # Map of entity ID to its node
    cycles: List[List[Any]] = Field(default_factory=list)      # List of detected cycles
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
        
    def build_graph(self, root_entity: Any, 
                    is_entity_check: Optional[Callable[[Any], bool]] = None,
                    get_entity_id: Optional[Callable[[Any], Any]] = None) -> CycleStatus:
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
            def is_entity_func(obj: Any) -> bool:
                return hasattr(obj, "ecs_id")
            is_entity_check_func = is_entity_func  
        else:
            is_entity_check_func = is_entity_check
                
        # Default ID function
        if get_entity_id is None:
            def get_entity_id_func(entity: Any) -> Any:
                return getattr(entity, "ecs_id", id(entity))
            entity_id_func = get_entity_id_func
        else:
            entity_id_func = get_entity_id
                
        # Clear existing graph
        self.nodes.clear()
        self.cycles.clear()
        
        # First, collect all entities and their dependencies
        entities_to_process = [root_entity]
        entity_dependencies: Dict[Any, Set[Any]] = {}  # Map entity ID to set of dependency IDs
        
        # Collect all entities and their immediate dependencies
        while entities_to_process:
            entity = entities_to_process.pop(0)
            entity_id = entity_id_func(entity)
            
            # Skip if already processed
            if entity_id in entity_dependencies:
                continue
                
            # Add to graph
            if entity_id not in self.nodes:
                self.nodes[entity_id] = GraphNode(entity=entity, entity_id=entity_id)
            
            # Find immediate dependencies
            deps = self._find_entity_references(entity, is_entity_check_func, entity_id_func)
            dep_ids = set()
            
            for dep_entity, _ in deps:
                dep_id = entity_id_func(dep_entity)
                dep_ids.add(dep_id)
                
                # Create node if needed
                if dep_id not in self.nodes:
                    self.nodes[dep_id] = GraphNode(entity=dep_entity, entity_id=dep_id)
                
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
        visited: Set[Any] = set()  # Nodes we've fully processed
        path: Set[Any] = set()     # Nodes in current path
        
        def find_cycles(node_id: Any) -> None:
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
            self.nodes[entity_id] = GraphNode(entity=entity, entity_id=entity_id)
            
        # Add dependencies if provided
        if dependencies:
            for dep in dependencies:
                if dep is not None:
                    dep_id = getattr(dep, "ecs_id", id(dep))
                    
                    # Create node for dependency if needed
                    if dep_id not in self.nodes:
                        self.nodes[dep_id] = GraphNode(entity=dep, entity_id=dep_id)
                        
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
                             is_entity_check_func: Callable[[Any], bool], 
                             entity_id_func: Callable[[Any], Any]) -> List[Tuple[Any, str]]:
        """
        Find all entity references in an object.
        
        Args:
            obj: Object to inspect
            is_entity_check_func: Function to determine if an object is an entity
            entity_id_func: Function to get an entity's ID
            
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
                if is_entity_check_func(value):
                    logger.debug(f"Found entity in dict at key {key}: {value}")
                    results.append((value, f"{key}"))
                elif isinstance(value, (dict, list, tuple, set)):
                    # Recursively process containers
                    nested = self._find_entity_references(
                        value, is_entity_check_func, entity_id_func)
                    results.extend([(e, f"{key}.{p}") for e, p in nested])
                    
        elif isinstance(obj, (list, tuple, set)):
            for i, value in enumerate(obj):
                if is_entity_check_func(value):
                    logger.debug(f"Found entity in list at index {i}: {value}")
                    results.append((value, f"[{i}]"))
                elif isinstance(value, (dict, list, tuple, set)):
                    # Recursively process containers
                    nested = self._find_entity_references(
                        value, is_entity_check_func, entity_id_func)
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
                    if value is not None and is_entity_check_func(value):
                        logger.debug(f"Found entity in attribute {attr_name}: {value}")
                        results.append((value, attr_name))
                    elif isinstance(value, (dict, list, tuple, set)):
                        # Recursively process containers
                        nested = self._find_entity_references(
                            value, is_entity_check_func, entity_id_func)
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
        depths: Dict[Any, int] = {}
        
        def calculate_depth(node_id: Any, path: Optional[Set[Any]] = None) -> int:
            """Calculate maximum dependency depth for a node."""
            if path is None:
                path = set()
                
            # Check for cycles - if we've seen this node before in this path
            if node_id in path:
                return 0
                
            # If already calculated, return cached value
            if node_id in depths:
                return depths[node_id]
                
            path_copy = path.copy()
            path_copy.add(node_id)
            
            # If no dependencies, depth is 0
            node = self.nodes.get(node_id)
            if not node or not node.dependencies:
                depths[node_id] = 0
                return 0
                
            # Calculate maximum depth of dependencies
            max_depth = 0
            for dep_id in node.dependencies:
                if dep_id in self.nodes:
                    depth = calculate_depth(dep_id, path_copy) + 1
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