"""
Simplified version of entity.py for better debugging experience.
Only includes core functionality, with SQL models simplified or removed.
"""
import sys
import json
import inspect
import logging
from typing import (
    Any, Dict, Optional, Type, TypeVar, List, Protocol, runtime_checkable,
    Union, Callable, get_args, cast, Self, Set, Tuple, Generic, get_origin
)
from uuid import UUID, uuid4
from datetime import datetime
from copy import deepcopy
from functools import wraps

from pydantic import BaseModel, Field, model_validator

# Import EntityDependencyGraph
from entity_dependencies import EntityDependencyGraph, CycleStatus

# Make sure EntityRegistry is available in __main__
# This is how we fix the import issue
EntityRegistry = None  # Will be defined at the end

##############################
# 1) The base_registry
##############################
class BaseRegistry:
    """
    A minimal base class that just has a get_registry_status method
    or any other common logic. We rely on the subclass to store
    the actual `_storage`.
    """
    @classmethod
    def get_registry_status(cls) -> Dict[str, Any]:
        return {"base_registry": True}

##############################
# 2) Type Definitions
##############################

# Define type variables
T = TypeVar('T')
SQMT = TypeVar('SQMT', bound='SQLModelType')
T_Self = TypeVar('T_Self', bound='Entity')

# Type for SQL model classes that implement to_entity/from_entity
class SQLModelType(Protocol):
    @classmethod
    def from_entity(cls, entity: 'Entity') -> 'SQLModelType': ...
    def to_entity(self) -> 'Entity': ...
    ecs_id: UUID
    lineage_id: UUID

##############################
# 3) Core comparison and storage utilities
##############################

def compare_entity_fields(
    entity1: Any, 
    entity2: Any, 
    exclude_fields: Optional[Set[str]] = None
) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Improved comparison method for entity fields.
    Compares content rather than references for better change detection.
    
    Returns:
        Tuple of (has_modifications, field_diffs_dict)
    """
    logger = logging.getLogger("EntityComparison")
    logger.info(f"Comparing entities: {type(entity1).__name__}({entity1.ecs_id}) vs {type(entity2).__name__}({entity2.ecs_id})")
    
    if exclude_fields is None:
        # Make sure we exclude all technical and implementation fields
        exclude_fields = {
            'id', 'ecs_id', 'created_at', 'parent_id', 'live_id', 
            'old_ids', 'lineage_id', 'from_storage', 'force_parent_fork', 
            'sql_root', 'deps_graph', 'is_being_registered'
        }
    
    # Get field sets for both entities
    entity1_fields = set(entity1.model_fields.keys()) - exclude_fields
    entity2_fields = set(entity2.model_fields.keys()) - exclude_fields
    
    logger.debug(f"Comparing {len(entity1_fields)} fields after excluding implementation fields")
    
    # Quick check for field set differences
    if entity1_fields != entity2_fields:
        diff_fields = entity1_fields.symmetric_difference(entity2_fields)
        logger.info(f"Schema difference detected: fields {diff_fields} differ between entities")
        return True, {f: {"type": "schema_change"} for f in diff_fields}
    
    # Detailed field comparison
    field_diffs = {}
    has_diffs = False
    
    # Check fields in first entity
    for field in entity1_fields:
        value1 = getattr(entity1, field)
        value2 = getattr(entity2, field)
        
        # If both are entities, compare by ecs_id instead of instance
        if isinstance(value1, Entity) and isinstance(value2, Entity):
            if value1.ecs_id != value2.ecs_id:
                has_diffs = True
                logger.info(f"Field '{field}' contains different entities: {value1.ecs_id} vs {value2.ecs_id}")
                field_diffs[field] = {
                    "type": "modified",
                    "old_id": str(value2.ecs_id),
                    "new_id": str(value1.ecs_id)
                }
        # If both are lists, compare items individually
        elif isinstance(value1, list) and isinstance(value2, list):
            if len(value1) != len(value2):
                has_diffs = True
                logger.info(f"Field '{field}' has different list lengths: {len(value1)} vs {len(value2)}")
                field_diffs[field] = {
                    "type": "modified",
                    "old_length": len(value2),
                    "new_length": len(value1)
                }
            else:
                # For lists of entities, compare by ecs_id
                if all(isinstance(v, Entity) for v in value1) and all(isinstance(v, Entity) for v in value2):
                    ids1 = {e.ecs_id for e in value1}
                    ids2 = {e.ecs_id for e in value2}
                    if ids1 != ids2:
                        has_diffs = True
                        logger.info(f"Field '{field}' contains lists of different entities")
                        logger.debug(f"List 1 IDs: {ids1}")
                        logger.debug(f"List 2 IDs: {ids2}")
                        field_diffs[field] = {
                            "type": "modified",
                            "old_ids": [str(id) for id in ids2],
                            "new_ids": [str(id) for id in ids1]
                        }
                # Otherwise compare normally
                elif value1 != value2:
                    has_diffs = True
                    logger.info(f"Field '{field}' has different list contents")
                    field_diffs[field] = {
                        "type": "modified",
                        "old": value2,
                        "new": value1
                    }
        # For all other types, compare normally
        elif value1 != value2:
            has_diffs = True
            logger.info(f"Field '{field}' has different values: {value1} vs {value2}")
            field_diffs[field] = {
                "type": "modified",
                "old": value2,
                "new": value1
            }
    
    logger.info(f"Comparison result: has_diffs={has_diffs}, found {len(field_diffs)} different fields")
    return has_diffs, field_diffs

def create_cold_snapshot(entity: 'Entity') -> 'Entity':
    """
    Create a cold snapshot of an entity for storage.
    Ensures proper deep copying and field preservation.
    """
    # Create deep copy first
    snapshot = deepcopy(entity)
    
    # Ensure from_storage flag is unset
    snapshot.from_storage = False
    
    # Return the snapshot
    return snapshot

##############################
# 4) The Entity + Diff
##############################

@runtime_checkable
class HasID(Protocol):
    """Protocol requiring an `ecs_id: UUID` field."""
    ecs_id: UUID

class EntityDiff:
    """Represents structured differences between entities."""
    def __init__(self) -> None:
        self.field_diffs: Dict[str, Dict[str, Any]] = {}

    def add_diff(self, field: str, diff_type: str, old_value: Any = None, new_value: Any = None) -> None:
        self.field_diffs[field] = {
            "type": diff_type,
            "old": old_value,
            "new": new_value
        }
        
    @classmethod
    def from_diff_dict(cls, diff_dict: Dict[str, Dict[str, Any]]) -> 'EntityDiff':
        """Create an EntityDiff from a difference dictionary."""
        diff = cls()
        diff.field_diffs = diff_dict
        return diff

    def has_changes(self) -> bool:
        """Check if there are any significant differences that require forking."""
        logger = logging.getLogger("EntityDiff")
        
        # Empty diffs = no changes
        if not self.field_diffs:
            logger.debug("No field differences found")
            return False
            
        logger.info(f"Checking significance of {len(self.field_diffs)} field differences")
        
        # For each field, check if it's a significant change (not just implementation details)
        for field, diff_info in self.field_diffs.items():
            # Skip implementation fields that don't need to trigger a new version
            if field in {
                'id', 'ecs_id', 'live_id', 'from_storage', 'force_parent_fork', 
                'sql_root', 'created_at', 'parent_id', 'old_ids', 'lineage_id'
            }:
                logger.debug(f"Field '{field}' is an implementation detail - not significant")
                continue
                
            # Any other field change is significant
            logger.info(f"Field '{field}' has significant changes (type: {diff_info.get('type', 'unknown')})")
            return True
            
        # No significant changes found
        logger.info("No significant changes found - all differences are implementation details")
        return False

class Entity(BaseModel):
    """
    Base class for registry-integrated, serializable entities with versioning support.

    Entity Lifecycle and Behavior:

    1. ENTITY CREATION AND REGISTRATION:
       - Create entity in memory
       - Initialize dependency graph for all sub-entities
       - Register root entity:
         a) Create cold snapshot
         b) Store in registry/SQL
         c) Done

    2. TRACED FUNCTION BEHAVIOR:
       - Get stored version by ID (ORM relationships automatically load the complete entity tree)
       - Compare current (warm) vs stored
       - If different: FORK

    3. FORKING PROCESS (Bottom-Up):
       - Compare with stored version (already complete with all nested entities)
       - If different:
         a) Fork entity (new ID, set parent)
         b) Store cold snapshot
         c) Update references in memory

    Key Principles:
    - Use explicit dependency graph to manage entity relationships
    - Bottom-up processing using topological sort from dependency graph
    - Only root entities handle registration to avoid circular dependencies
    - Proper handling of circular references without changing object model

    Attributes:
        id: Unique identifier for this version
        live_id: Identifier for the "warm" copy in memory
        created_at: Timestamp of creation
        parent_id: ID of the previous version
        lineage_id: ID grouping all versions of this entity
        old_ids: Historical list of previous IDs
        from_storage: Whether this instance was loaded from storage
        force_parent_fork: Flag indicating nested changes requiring parent fork
    """
    ecs_id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    live_id: UUID = Field(default_factory=uuid4, description="Live/warm identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    parent_id: Optional[UUID] = None
    lineage_id: UUID = Field(default_factory=uuid4)
    old_ids: List[UUID] = Field(default_factory=list)
    from_storage: bool = Field(default=False, description="Whether the entity was loaded from storage")
    force_parent_fork: bool = Field(default=False, description="Internal flag to force parent forking")
    sql_root: bool = Field(default=False, description="Whether the entity is the root of an SQL entity tree")
    # Dependency graph - not serialized, transient state
    deps_graph: Optional[Any] = Field(default=None, exclude=True)
    # Flag to prevent recursive registration
    is_being_registered: bool = Field(default=False, exclude=True)
    
    model_config = {
        "json_encoders": {
            UUID: str,
            datetime: lambda dt: dt.isoformat()
        }
    }

    def __hash__(self) -> int:
        """Make entity hashable by combining id and live_id."""
        return hash((self.ecs_id, self.live_id))

    def __eq__(self, other: object) -> bool:
        """Entities are equal if they have the same id and live_id."""
        if not isinstance(other, Entity):
            return NotImplemented
        return self.ecs_id == other.ecs_id and self.live_id == other.live_id
        
    def __repr__(self) -> str:
        """
        Custom representation that avoids circular references.
        Only shows the entity type and ID for a cleaner representation.
        """
        return f"{type(self).__name__}({str(self.ecs_id)})"
        
    def initialize_deps_graph(self) -> None:
        """
        Initialize dependency graph for this entity and all sub-entities.
        Creates a new graph and builds entity dependencies.
        """
        logger = logging.getLogger("EntityDepsGraph")
        logger.info(f"Initializing dependency graph for {type(self).__name__}({self.ecs_id})")
        
        # Create new graph
        self.deps_graph = EntityDependencyGraph()
        
        # Build the graph
        status = self.deps_graph.build_graph(self)
        
        # Share the graph with all sub-entities
        for sub in self.get_sub_entities():
            sub.deps_graph = self.deps_graph
            
        # Log cycle detection
        if status == CycleStatus.CYCLE_DETECTED:
            cycles = self.deps_graph.get_cycles()
            logger.warning(f"Detected {len(cycles)} cycles in entity relationships")
        else:
            logger.info(f"No cycles detected in entity relationships")
            
    def is_root_entity(self) -> bool:
        """
        Determine if this entity is a root entity.
        A root entity is one that should handle its own registration.
        """
        # SQL mode: only sql_root entities are considered roots
        storage_info = EntityRegistry.get_registry_status()
        using_sql_storage = storage_info.get('storage') == 'sql'
        if using_sql_storage:
            return self.sql_root
        
        # In-memory mode: for now, consider all entities as roots for simplicity
        # This is a temporary solution until we fully resolve circular references
        return True

    @model_validator(mode='after')
    def register_on_create(self) -> Self:
        """Register this entity when it's created."""
        # Check if EntityRegistry exists in __main__
        if 'EntityRegistry' not in globals() or EntityRegistry is None:
            # Skip registration if EntityRegistry isn't defined yet
            return self
        
        # Skip if already being registered or from storage
        if self.is_being_registered or self.from_storage:
            return self
            
        # Initialize dependency graph if not already done
        if not hasattr(self, 'deps_graph') or self.deps_graph is None:
            self.initialize_deps_graph()
            
        # Only register root entities to avoid circular references
        if self.is_root_entity():
            try:
                # Mark as being registered to prevent recursion
                self.is_being_registered = True
                
                # Mark all sub-entities as being registered
                for sub in self.get_sub_entities():
                    sub.is_being_registered = True
                    
                # Register through EntityRegistry
                EntityRegistry.register(self)
            finally:
                # Clean up flags
                self.is_being_registered = False
                for sub in self.get_sub_entities():
                    sub.is_being_registered = False
        
        return self

    def fork(self: T_Self) -> T_Self:
        """
        Fork this entity if it differs from its stored version.
        Uses the dependency graph for efficient handling of circular references.
        
        The forking process follows these steps:
        1. Get the stored version of this entity
        2. Check for modifications in the entire entity tree using the dependency graph
        3. Use topological sort from dependency graph for bottom-up processing
        4. Fork each entity in dependency order
        5. Register each forked entity with storage
        6. Return the forked entity
        """
        logger = logging.getLogger("EntityFork")
        logger.info(f"Forking entity: {type(self).__name__}({self.ecs_id})")
        
        # Get stored version
        frozen = EntityRegistry.get_cold_snapshot(self.ecs_id)
        if frozen is None:
            logger.info(f"No stored version found for {self.ecs_id}, skipping fork")
            return self
            
        # Initialize dependency graph if not already done
        if not hasattr(self, 'deps_graph') or self.deps_graph is None:
            self.initialize_deps_graph()
            
        # Check what needs to be forked
        needs_fork, entities_to_fork = self.has_modifications(frozen)
        if not needs_fork:
            logger.info(f"No modifications detected, skipping fork")
            return self
            
        logger.info(f"Found {len(entities_to_fork)} entities to fork")
            
        # Create a new dependency graph for the entities to fork
        # This ensures we only process the entities that need forking
        fork_graph = EntityDependencyGraph()
        
        # Build a sub-graph containing only entities that need forking
        for entity in entities_to_fork:
            # Add dependencies from the original graph
            deps = []
            if entity.deps_graph:
                node = entity.deps_graph.get_node(entity.ecs_id)
                if node:
                    deps = [entity.deps_graph.find_entity_by_id(dep_id) for dep_id in node.dependencies]
                    deps = [dep for dep in deps if dep and dep in entities_to_fork]
            
            # Add entity and its dependencies to fork graph
            fork_graph.add_entity(entity, deps)
        
        # Get entities in topological order (dependencies first)
        sorted_entities = fork_graph.get_topological_sort()
        logger.info(f"Sorted {len(sorted_entities)} entities for forking in dependency order")
        
        # Fork entities in dependency order (bottom-up)
        id_map = {}  # Map old IDs to new entities
        for entity in sorted_entities:
            old_id = entity.ecs_id
            entity.ecs_id = uuid4()
            entity.parent_id = old_id
            entity.old_ids.append(old_id)
            id_map[old_id] = entity
            logger.debug(f"Forked entity {type(entity).__name__}: {old_id} -> {entity.ecs_id}")
            
            # Update references in parent entities
            for parent in sorted_entities:
                if parent == entity:
                    continue
                    
                # Update references in parent's fields
                for field_name, field_info in parent.model_fields.items():
                    value = getattr(parent, field_name)
                    if value is None:
                        continue
                        
                    # Direct entity reference
                    if isinstance(value, Entity) and value.ecs_id == old_id:
                        setattr(parent, field_name, entity)
                        logger.debug(f"Updated reference in {parent.ecs_id}.{field_name}")
                        
                    # List/tuple of entities
                    elif isinstance(value, (list, tuple)):
                        if isinstance(value, tuple):
                            value = list(value)
                        for i, item in enumerate(value):
                            if isinstance(item, Entity) and item.ecs_id == old_id:
                                value[i] = entity
                                logger.debug(f"Updated reference in {parent.ecs_id}.{field_name}[{i}]")
                        if isinstance(getattr(parent, field_name), tuple):
                            value = tuple(value)
                        setattr(parent, field_name, value)
                        
                    # Dict containing entities
                    elif isinstance(value, dict):
                        for k, v in value.items():
                            if isinstance(v, Entity) and v.ecs_id == old_id:
                                value[k] = entity
                                logger.debug(f"Updated reference in {parent.ecs_id}.{field_name}[{k}]")
                        setattr(parent, field_name, value)
            
            # Mark as not being registered to avoid recursion issues
            entity.is_being_registered = True
            
            # Register the forked entity with storage
            EntityRegistry.register(entity)
            
            # Reset flag
            entity.is_being_registered = False
        
        # Update the dependency graph for the forked entities
        self.initialize_deps_graph()
        
        logger.info(f"Fork complete: {type(self).__name__}({self.ecs_id})")
        # Return the forked entity
        return self

    def has_modifications(self, other: "Entity") -> Tuple[bool, Dict["Entity", EntityDiff]]:
        """
        Check if this entity or any nested entities differ from their stored versions.
        Uses dependency graph for handling circular references and bottom-up traversal.
        
        Returns:
            Tuple of (any_changes, {changed_entity: its_changes})
        """
        # Get storage type to adjust comparison strictness
        storage_info = EntityRegistry.get_registry_status()
        using_sql_storage = storage_info.get('storage') == 'sql'
        
        logger = logging.getLogger("EntityModification")
        logger.info(f"Checking modifications: {type(self).__name__}({self.ecs_id}) vs {type(other).__name__}({other.ecs_id}) [SQL: {using_sql_storage}]")
        
        modified_entities: Dict["Entity", EntityDiff] = {}
        
        # Early exit if comparing an entity with itself
        # only if we are using sql storage
        if self.ecs_id == other.ecs_id and self.live_id == other.live_id and using_sql_storage:
            logger.debug(f"Same entity instance detected (same ecs_id and live_id): {self.ecs_id}")
            return False, {}
            
        # Initialize dependency graph if not already done
        if not hasattr(self, 'deps_graph') or self.deps_graph is None:
            self.initialize_deps_graph()
        
        # Initialize dependency graph for the other entity too
        if not hasattr(other, 'deps_graph') or other.deps_graph is None:
            other.initialize_deps_graph()
            
        # Get all entities in topological order (dependencies first)
        # This ensures bottom-up processing
        sorted_entities = self.deps_graph.get_topological_sort()
        
        # Create lookup for other entity's sub-entities by ID
        other_entities = {e.ecs_id: e for e in other.get_sub_entities()}
        other_entities[other.ecs_id] = other  # Include the root entity
        
        # Process entities in topological order (bottom-up)
        for entity in sorted_entities:
            # Find matching entity in other tree
            if entity.ecs_id not in other_entities:
                logger.debug(f"Entity {entity.ecs_id} not found in other tree, skipping")
                continue
                
            other_entity = other_entities[entity.ecs_id]
            
            # Compare direct fields
            has_diffs, field_diffs = compare_entity_fields(entity, other_entity)
            
            if has_diffs:
                logger.info(f"Direct field differences found in {type(entity).__name__}({entity.ecs_id})")
                significant_changes = False
                
                # Define implementation fields to skip
                implementation_fields = {
                    'id', 'ecs_id', 'live_id', 'created_at', 'parent_id', 
                    'old_ids', 'lineage_id', 'from_storage', 'force_parent_fork', 
                    'sql_root', 'deps_graph', 'is_being_registered'
                }
                
                for field, diff_info in field_diffs.items():
                    # Skip implementation fields 
                    if field in implementation_fields:
                        logger.debug(f"Skipping implementation field '{field}'")
                        continue
                        
                    # Any other field difference is significant
                    logger.info(f"Field '{field}' has significant changes")
                    significant_changes = True
                    break
                            
                if significant_changes:
                    logger.info(f"Entity {type(entity).__name__}({entity.ecs_id}) has significant changes requiring fork")
                    # If we already have an empty diff (from nested changes), update it
                    if entity in modified_entities:
                        modified_entities[entity].field_diffs.update(field_diffs)
                    else:
                        modified_entities[entity] = EntityDiff.from_diff_dict(field_diffs)
                        
                    # Mark parents for forking too
                    parent_ids = self.deps_graph.get_dependent_ids(entity.ecs_id)
                    for parent_id in parent_ids:
                        parent = self.deps_graph.find_entity_by_id(parent_id)
                        if parent and parent not in modified_entities:
                            logger.info(f"Marking parent {type(parent).__name__}({parent.ecs_id}) for forking due to child changes")
                            modified_entities[parent] = EntityDiff()
        
        needs_fork = bool(modified_entities)
        logger.info(f"Modification check result: needs_fork={needs_fork}, modified_entities={len(modified_entities)}")
        return needs_fork, modified_entities

    def compute_diff(self, other: "Entity") -> EntityDiff:
        """Compute detailed differences between this entity and another entity."""
        _, field_diffs = compare_entity_fields(self, other)
        return EntityDiff.from_diff_dict(field_diffs)

    def entity_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Skip versioning fields, plus recursively dump nested Entities.
        """
        exclude_keys = set(kwargs.get('exclude', set()))
        # Ensure we exclude both old and new field names and all implementation details
        exclude_keys |= {
            'id', 'ecs_id', 'created_at', 'parent_id', 'live_id', 
            'old_ids', 'lineage_id', 'from_storage', 'force_parent_fork', 
            'sql_root'
        }
        kwargs['exclude'] = exclude_keys

        data = super().model_dump(*args, **kwargs)
        for k, v in data.items():
            if isinstance(v, Entity):
                data[k] = v.entity_dump()
            elif isinstance(v, list):
                new_list = []
                for item in v:
                    if isinstance(item, Entity):
                        new_list.append(item.entity_dump())
                    else:
                        new_list.append(item)
                data[k] = new_list
        return data

    @classmethod
    def get(cls: Type["Entity"], entity_id: UUID) -> Optional["Entity"]:
        """Get an entity from the registry by ID."""
        ent = EntityRegistry.get(entity_id, expected_type=cls)
        return cast(Optional["Entity"], ent)

    @classmethod
    def list_all(cls: Type["Entity"]) -> List["Entity"]:
        """List all entities of this type."""
        return EntityRegistry.list_by_type(cls)

    @classmethod
    def get_many(cls: Type["Entity"], ids: List[UUID]) -> List["Entity"]:
        """Get multiple entities by ID."""
        return EntityRegistry.get_many(ids, expected_type=cls)

    def get_sub_entities(self, visited: Optional[Set[UUID]] = None) -> Set['Entity']:
        """
        Get all sub-entities of this entity.
        Uses a visited set to prevent circular reference recursion.
        
        Args:
            visited: Optional set of entity IDs that have already been visited
            
        Returns:
            Set of sub-entities
        """
        # Initialize visited set if not provided
        if visited is None:
            visited = set()
            
        # Skip if we've already visited this entity
        if self.ecs_id in visited:
            return set()
            
        # Add self to visited
        visited.add(self.ecs_id)
        
        # Collect all sub-entities
        nested: Set['Entity'] = set()
        
        for field_name, field_info in self.model_fields.items():
            # Skip implementation fields that might contain references to parent entities
            if field_name in {'deps_graph', 'is_being_registered', 'parent', 'parent_id'}:
                continue
                
            value = getattr(self, field_name)
            if value is None:
                continue
                
            # Handle lists of entities
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, Entity) and item.ecs_id not in visited:
                        nested.add(item)
                        # Pass along the visited set to avoid cycles
                        nested.update(item.get_sub_entities(visited))
            
            # Handle direct entity references
            elif isinstance(value, Entity) and value.ecs_id not in visited:
                nested.add(value)
                # Pass along the visited set to avoid cycles
                nested.update(value.get_sub_entities(visited))
                
        return nested

##############################
# 5) Storage Protocol
##############################

class EntityStorage(Protocol):
    """
    Generic interface for storing and retrieving Entities, building lineage, etc.
    """
    def has_entity(self, entity_id: UUID) -> bool: ...
    def get_cold_snapshot(self, entity_id: UUID) -> Optional[Entity]: ...
    def register(self, entity_or_id: Union[Entity, UUID]) -> Optional[Entity]: ...
    def get(self, entity_id: UUID, expected_type: Optional[Type[Entity]]) -> Optional[Entity]: ...
    def list_by_type(self, entity_type: Type[Entity]) -> List[Entity]: ...
    def get_many(self, entity_ids: List[UUID], expected_type: Optional[Type[Entity]]) -> List[Entity]: ...
    def get_registry_status(self) -> Dict[str, Any]: ...
    def set_inference_orchestrator(self, orchestrator: object) -> None: ...
    def get_inference_orchestrator(self) -> Optional[object]: ...
    def clear(self) -> None: ...
    def get_lineage_entities(self, lineage_id: UUID) -> List[Entity]: ...
    def has_lineage_id(self, lineage_id: UUID) -> bool: ...
    def get_lineage_ids(self, lineage_id: UUID) -> List[UUID]: ...


class InMemoryEntityStorage(EntityStorage):
    """
    In-memory storage using Python's object references.
    """
    def __init__(self) -> None:
        self._logger = logging.getLogger("InMemoryEntityStorage")
        self._registry: Dict[UUID, Entity] = {}
        self._entity_class_map: Dict[UUID, Type[Entity]] = {}
        self._lineages: Dict[UUID, List[UUID]] = {}
        self._inference_orchestrator: Optional[object] = None

    def has_entity(self, entity_id: UUID) -> bool:
        """Check if entity exists in storage."""
        return entity_id in self._registry

    def get_cold_snapshot(self, entity_id: UUID) -> Optional[Entity]:
        """Get the cold (stored) version of an entity."""
        return self._registry.get(entity_id)

    def register(self, entity_or_id: Union[Entity, UUID]) -> Optional[Entity]:
        """Register an entity or retrieve it by ID."""
        if isinstance(entity_or_id, UUID):
            return self.get(entity_or_id, None)

        entity = entity_or_id
            
        # Check if entity already exists
        if self.has_entity(entity.ecs_id):
            # Get existing version
            existing = self.get_cold_snapshot(entity.ecs_id)
            if existing and entity.has_modifications(existing):
                # Fork the entity and all its nested entities
                entity = entity.fork()
        
        # Collect all entities that need to be stored
        entities_to_store: Dict[UUID, Entity] = {}
        
        def collect_entities(e: Entity) -> None:
            if e.ecs_id not in entities_to_store:
                # Create a cold snapshot for storage
                snap = create_cold_snapshot(e)
                entities_to_store[e.ecs_id] = snap
                # Collect nested entities
                for sub in e.get_sub_entities():
                    collect_entities(sub)
        
        # Collect all entities in the tree
        collect_entities(entity)
        
        # Store all entities
        for e in entities_to_store.values():
            self._store_cold_snapshot(e)
            
        return entity

    def get(self, entity_id: UUID, expected_type: Optional[Type[Entity]] = None) -> Optional[Entity]:
        """Get an entity by ID with optional type checking."""
        ent = self._registry.get(entity_id)
        if not ent:
            return None
            
        if expected_type and not isinstance(ent, expected_type):
            self._logger.error(f"Type mismatch: got {type(ent).__name__}, expected {expected_type.__name__}")
            return None
            
        # Create a warm copy
        warm_copy = deepcopy(ent)
        warm_copy.live_id = uuid4()
        warm_copy.from_storage = True  # Mark as coming from storage
        
        return warm_copy

    def list_by_type(self, entity_type: Type[Entity]) -> List[Entity]:
        """List all entities of a specific type."""
        return [
            deepcopy(e)
            for e in self._registry.values()
            if isinstance(e, entity_type)
        ]

    def get_many(self, entity_ids: List[UUID], expected_type: Optional[Type[Entity]] = None) -> List[Entity]:
        """Get multiple entities by ID."""
        return [e for eid in entity_ids if (e := self.get(eid, expected_type)) is not None]

    def get_registry_status(self) -> Dict[str, Any]:
        """Get status information about the registry."""
        return {
            "storage": "in_memory",
            "in_memory": True,
            "entity_count": len(self._registry),
            "lineage_count": len(self._lineages)
        }

    def set_inference_orchestrator(self, orchestrator: object) -> None:
        """Set an inference orchestrator."""
        self._inference_orchestrator = orchestrator

    def get_inference_orchestrator(self) -> Optional[object]:
        """Get the current inference orchestrator."""
        return self._inference_orchestrator

    def clear(self) -> None:
        """Clear all data from storage."""
        self._registry.clear()
        self._entity_class_map.clear()
        self._lineages.clear()

    def get_lineage_entities(self, lineage_id: UUID) -> List[Entity]:
        """Get all entities with a specific lineage ID."""
        return [e for e in self._registry.values() if e.lineage_id == lineage_id]

    def has_lineage_id(self, lineage_id: UUID) -> bool:
        """Check if a lineage ID exists."""
        return any(e for e in self._registry.values() if e.lineage_id == lineage_id)

    def get_lineage_ids(self, lineage_id: UUID) -> List[UUID]:
        """Get all entity IDs with a specific lineage ID."""
        return [e.ecs_id for e in self._registry.values() if e.lineage_id == lineage_id]

    def _store_cold_snapshot(self, entity: Entity) -> None:
        """Store a cold snapshot of an entity and all its sub-entities."""
        stored = set()
        
        def store_recursive(e: Entity) -> None:
            if e in stored:  # Using new hash/eq
                return
                
            stored.add(e)
            snap = create_cold_snapshot(e)
            self._registry[e.ecs_id] = snap
            
            # Update lineage tracking
            if e.lineage_id not in self._lineages:
                self._lineages[e.lineage_id] = []
            if e.ecs_id not in self._lineages[e.lineage_id]:
                self._lineages[e.lineage_id].append(e.ecs_id)
                
            # Store all sub-entities
            for sub in e.get_sub_entities():
                store_recursive(sub)
                
        store_recursive(entity)

# SQL implementation removed/simplified for clarity

##############################
# 7) Registry Facade
##############################

class EntityRegistry(BaseRegistry):
    """
    Improved static registry class with unified lineage handling.
    """
    _logger = logging.getLogger("EntityRegistry")
    _storage: EntityStorage = InMemoryEntityStorage()  # default

    @classmethod
    def use_storage(cls, storage: EntityStorage) -> None:
        """Set the storage implementation to use."""
        cls._storage = storage
        cls._logger.info(f"Now using {type(storage).__name__} for storage")

    # Simple delegation methods
    @classmethod
    def has_entity(cls, entity_id: UUID) -> bool:
        return cls._storage.has_entity(entity_id)

    @classmethod
    def get_cold_snapshot(cls, entity_id: UUID) -> Optional[Entity]:
        return cls._storage.get_cold_snapshot(entity_id)

    @classmethod
    def register(cls, entity_or_id: Union[Entity, UUID]) -> Optional[Entity]:
        return cls._storage.register(entity_or_id)

    @classmethod
    def get(cls, entity_id: UUID, expected_type: Optional[Type[Entity]] = None) -> Optional[Entity]:
        return cls._storage.get(entity_id, expected_type)

    @classmethod
    def list_by_type(cls, entity_type: Type[Entity]) -> List[Entity]:
        return cls._storage.list_by_type(entity_type)

    @classmethod
    def get_many(cls, entity_ids: List[UUID], expected_type: Optional[Type[Entity]] = None) -> List[Entity]:
        return cls._storage.get_many(entity_ids, expected_type)

    @classmethod
    def get_registry_status(cls) -> Dict[str, Any]:
        """Get combined status from base registry and storage."""
        base = super().get_registry_status()
        store = cls._storage.get_registry_status()
        return {**base, **store}

    @classmethod
    def set_inference_orchestrator(cls, orchestrator: object) -> None:
        cls._storage.set_inference_orchestrator(orchestrator)

    @classmethod
    def get_inference_orchestrator(cls) -> Optional[object]:
        return cls._storage.get_inference_orchestrator()

    @classmethod
    def clear(cls) -> None:
        cls._storage.clear()

    # Lineage methods
    @classmethod
    def has_lineage_id(cls, lineage_id: UUID) -> bool:
        return cls._storage.has_lineage_id(lineage_id)

    @classmethod
    def get_lineage_ids(cls, lineage_id: UUID) -> List[UUID]:
        return cls._storage.get_lineage_ids(lineage_id)
        
    @classmethod
    def get_lineage_entities(cls, lineage_id: UUID) -> List[Entity]:
        """Get all entities with a specific lineage ID."""
        return cls._storage.get_lineage_entities(lineage_id)

    @classmethod
    def build_lineage_tree(cls, lineage_id: UUID) -> Dict[UUID, Dict[str, Any]]:
        """Build a hierarchical tree from lineage entities."""
        nodes = cls.get_lineage_entities(lineage_id)
        if not nodes:
            return {}

        # Index entities by ID
        by_id = {e.ecs_id: e for e in nodes}
        
        # Find root entities (those without parents in this lineage)
        roots = [e for e in nodes if e.parent_id not in by_id]

        # Build tree structure
        tree: Dict[UUID, Dict[str, Any]] = {}

        def process_entity(entity: Entity, depth: int = 0) -> None:
            """Process a single entity and its children."""
            # Calculate differences from parent
            diff_from_parent = None
            if entity.parent_id and entity.parent_id in by_id:
                parent = by_id[entity.parent_id]
                diff = entity.compute_diff(parent)
                diff_from_parent = diff.field_diffs

            # Add to tree
            tree[entity.ecs_id] = {
                "entity": entity,
                "children": [],
                "depth": depth,
                "parent_id": entity.parent_id,
                "created_at": entity.created_at,
                "data": entity.entity_dump(),
                "diff_from_parent": diff_from_parent
            }

            # Link to parent
            if entity.parent_id and entity.parent_id in tree:
                tree[entity.parent_id]["children"].append(entity.ecs_id)

            # Process children
            children = [e for e in nodes if e.parent_id == entity.ecs_id]
            for child in children:
                process_entity(child, depth + 1)

        # Process all roots
        for root in roots:
            process_entity(root, 0)
            
        return tree

##############################
# 8) Entity Tracing - Simplified
##############################

def _collect_entities(args: tuple, kwargs: dict) -> Dict[int, Entity]:
    """Helper to collect all Entity instances from args and kwargs with their memory ids."""
    logger = logging.getLogger("EntityCollection")
    logger.debug(f"Collecting entities from {len(args)} args and {len(kwargs)} kwargs")
    
    entities = {}
    
    def scan(obj: Any, path: str = "") -> None:
        if isinstance(obj, Entity):
            entities[id(obj)] = obj
            logger.debug(f"Found entity {type(obj).__name__}({obj.ecs_id}) at path {path}")
        elif isinstance(obj, (list, tuple, set)):
            logger.debug(f"Scanning collection at path {path} with {len(obj)} items")
            for i, item in enumerate(obj):
                scan(item, f"{path}[{i}]")
        elif isinstance(obj, dict):
            logger.debug(f"Scanning dict at path {path} with {len(obj)} keys")
            for k, v in obj.items():
                scan(v, f"{path}.{k}")
    
    # Scan args and kwargs
    for i, arg in enumerate(args):
        scan(arg, f"args[{i}]")
    for key, arg in kwargs.items():
        scan(arg, f"kwargs[{key}]")
    
    logger.info(f"Collected {len(entities)} unique entities")
    return entities

def entity_tracer(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to trace entity modifications and handle versioning.
    Automatically detects and handles all Entity instances in arguments.
    Works with both sync and async functions, and both storage types.
    
    Simplified for clarity.
    """
    logger = logging.getLogger("EntityTracer")
    
    # Handle detection of async functions safely
    is_async = False
    try:
        import inspect as local_inspect
        is_async = local_inspect.iscoroutinefunction(func)
    except (ImportError, AttributeError):
        # Fallback method
        is_async = hasattr(func, '__await__') or (hasattr(func, '__code__') and func.__code__.co_flags & 0x80)
    
    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        # Collect all entities from inputs
        entities = _collect_entities(args, kwargs)
        
        # Check for modifications before call
        for entity_id, entity in entities.items():
            cold_snapshot = EntityRegistry.get_cold_snapshot(entity.ecs_id)
            if cold_snapshot:
                # Detect modifications
                needs_fork, _ = entity.has_modifications(cold_snapshot)
                if needs_fork:
                    # Fork before calling the function
                    entity.fork()

        # Call the function
        result = func(*args, **kwargs)

        # Check for modifications after call
        for entity_id, entity in entities.items():
            cold_snapshot = EntityRegistry.get_cold_snapshot(entity.ecs_id)
            if cold_snapshot:
                needs_fork, _ = entity.has_modifications(cold_snapshot)
                if needs_fork:
                    # Fork after calling the function
                    entity.fork()
                    
        # If result is an entity that was modified, return the forked version
        if isinstance(result, Entity) and id(result) in entities:
            return entities[id(result)]

        return result
    
    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        # Same logic as sync_wrapper but with async function call
        entities = _collect_entities(args, kwargs)
        
        for entity_id, entity in entities.items():
            cold_snapshot = EntityRegistry.get_cold_snapshot(entity.ecs_id)
            if cold_snapshot:
                needs_fork, _ = entity.has_modifications(cold_snapshot)
                if needs_fork:
                    entity.fork()

        # Call the async function
        result = await func(*args, **kwargs)

        for entity_id, entity in entities.items():
            cold_snapshot = EntityRegistry.get_cold_snapshot(entity.ecs_id)
            if cold_snapshot:
                needs_fork, _ = entity.has_modifications(cold_snapshot)
                if needs_fork:
                    entity.fork()
                    
        if isinstance(result, Entity) and id(result) in entities:
            return entities[id(result)]

        return result
    
    # Return appropriate wrapper
    return async_wrapper if is_async else sync_wrapper

# Now set the EntityRegistry variable globally
# This is to make it available for the Entity class to find it during registration
globals()['EntityRegistry'] = EntityRegistry