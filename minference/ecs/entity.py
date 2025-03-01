############################################################
# entity.py
############################################################

"""
Entity System with Hierarchical Version Control

This module implements a hierarchical entity system with version control capabilities.
Key concepts:

1. ENTITY IDENTITY AND STATE:
   - Each entity has both an `id` (version identifier) and a `live_id` (memory state identifier)
   - Entities are hashable by (id, live_id) combination to distinguish warm/cold copies
   - Cold snapshots are stored versions, warm copies are in-memory working versions

2. HIERARCHICAL STRUCTURE:
   - Entities can contain other entities (sub-entities) in fields, lists, or dictionaries
   - The `get_sub_entities()` method recursively discovers all nested entities
   - Changes in sub-entities trigger parent entity versioning

3. MODIFICATION DETECTION:
   - `has_modifications()` performs deep comparison of entity trees
   - Returns both whether changes exist and the set of modified entities
   - Handles nested changes automatically through recursive comparison

4. FORKING PROCESS:
   - When changes are detected, affected entities get new IDs
   - Parent entities automatically fork when sub-entities change
   - All changes happen in memory first, then are committed to storage
   - No explicit dependency tracking needed - hierarchy is discovered dynamically

5. STORAGE LAYER:
   - Stores complete entity trees in a single operation
   - Uses cold snapshots to preserve version history
   - Handles circular references and complex hierarchies automatically

Example Usage:
```python
# Create and modify an entity
entity = ComplexEntity(nested=SubEntity(...))
entity.nested.value = "new"

# Automatic versioning on changes
if entity.has_modifications(stored_version):
    new_version = entity.fork()  # Creates new IDs for changed entities
    
# Storage handles complete trees
EntityRegistry.register(new_version)  # Stores all sub-entities
```

Implementation Notes:
- No dependency graphs needed - hierarchy is discovered through type hints
- Warm/cold copy distinction through live_id
- Bottom-up change propagation through recursive detection
- Complete tree operations for consistency
"""

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
    Unified comparison method for entity fields.
    
    Returns:
        Tuple of (has_modifications, field_diffs_dict)
    """
    if exclude_fields is None:
        exclude_fields = {'id', 'created_at', 'parent_id', 'live_id', 'old_ids', 'lineage_id', 'from_storage'}
    
    # Get field sets for both entities
    entity1_fields = set(entity1.model_fields.keys()) - exclude_fields
    entity2_fields = set(entity2.model_fields.keys()) - exclude_fields
    
    # Quick check for field set differences
    if entity1_fields != entity2_fields:
        return True, {f: {"type": "schema_change"} for f in entity1_fields.symmetric_difference(entity2_fields)}
    
    # Detailed field comparison
    field_diffs = {}
    has_diffs = False
    
    # Check fields in first entity
    for field in entity1_fields:
        value1 = getattr(entity1, field)
        value2 = getattr(entity2, field)
        
        if value1 != value2:
            has_diffs = True
            field_diffs[field] = {
                "type": "modified",
                "old": value2,
                "new": value1
            }
    
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
        """Check if there are any differences."""
        return bool(self.field_diffs)

class Entity(BaseModel):
    """
    Base class for registry-integrated, serializable entities with versioning support.

    Entity Lifecycle and Behavior:

    1. ENTITY CREATION AND REGISTRATION:
       - Create entity in memory
       - Register:
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
    - Proper ORM relationships = complete entity trees on load
    - Single DB fetch gets entire entity structure
    - Memory state is already correct after forking
    - Bottom-up processing happens naturally through ORM relationships

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

    @model_validator(mode='after')
    def register_on_create(self) -> Self:
        """Register this entity when it's created."""
        from __main__ import EntityRegistry
        # Only register if:
        # 1. Not from storage
        # 2. AND is marked as a root entity
        if not self.from_storage and self.sql_root:
            EntityRegistry.register(self)
        return self

    def fork(self: T_Self) -> T_Self:
        """
        Fork this entity if it differs from its stored version.
        Works entirely in memory - database operations happen at registration.
        
        The forking process follows these steps:
        1. Get the stored version of this entity
        2. Check for modifications in the entire entity tree
        3. Sort entities by dependency (bottom-up) to ensure proper forking order
        4. Fork each entity in the correct order
        5. Register each forked entity with storage
        6. Return the forked entity
        """
        from __main__ import EntityRegistry
        
        # Get stored version
        frozen = EntityRegistry.get_cold_snapshot(self.ecs_id)
        if frozen is None:
            return self
            
        # Check what needs to be forked
        needs_fork, entities_to_fork = self.has_modifications(frozen)
        if not needs_fork:
            return self
            
        # Sort entities by dependency (bottom-up)
        sorted_entities = []
        processed = set()
        
        def process_entity(entity: Entity) -> None:
            if entity in processed:
                return
            # Process dependencies first
            for sub in entity.get_sub_entities():
                if sub in entities_to_fork and sub not in processed:
                    process_entity(sub)
            sorted_entities.append(entity)
            processed.add(entity)
        
        # Process all entities that need forking
        for entity in entities_to_fork:
            if entity not in processed:
                process_entity(entity)
        
        # Fork entities in dependency order (bottom-up)
        id_map = {}  # Map old IDs to new entities
        for entity in sorted_entities:
            old_id = entity.ecs_id
            entity.ecs_id = uuid4()
            entity.parent_id = old_id
            entity.old_ids.append(old_id)
            id_map[old_id] = entity
            
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
                        
                    # List/tuple of entities
                    elif isinstance(value, (list, tuple)):
                        if isinstance(value, tuple):
                            value = list(value)
                        for i, item in enumerate(value):
                            if isinstance(item, Entity) and item.ecs_id == old_id:
                                value[i] = entity
                        if isinstance(getattr(parent, field_name), tuple):
                            value = tuple(value)
                        setattr(parent, field_name, value)
                        
                    # Dict containing entities
                    elif isinstance(value, dict):
                        for k, v in value.items():
                            if isinstance(v, Entity) and v.ecs_id == old_id:
                                value[k] = entity
                        setattr(parent, field_name, value)
            
            # Register the forked entity with storage
            EntityRegistry.register(entity)
        
        # Return the forked entity
        return self

    def has_modifications(self, other: "Entity") -> Tuple[bool, Dict["Entity", EntityDiff]]:
        """
        Check if this entity or any nested entities differ from their stored versions.
        Returns:
            Tuple of (any_changes, {changed_entity: its_changes})
        """
        modified_entities: Dict["Entity", EntityDiff] = {}
        
        # Get all sub-entities first to ensure bottom-up processing
        my_subs = self.get_sub_entities()
        other_subs = other.get_sub_entities()
        
        # Check sub-entities first (bottom-up)
        for my_sub in my_subs:
            other_sub = next((e for e in other_subs if e.ecs_id == my_sub.ecs_id), None)
            if other_sub:
                # Recursively check nested entity
                sub_changed, sub_modified = my_sub.has_modifications(other_sub)
                if sub_changed:
                    # If a nested entity changed, add it and its modifications
                    modified_entities.update(sub_modified)
                    # Mark parent as needing fork (but without direct changes)
                    if self not in modified_entities:
                        modified_entities[self] = EntityDiff()
        
        # Then check direct fields (after sub-entities)
        has_diffs, field_diffs = compare_entity_fields(self, other)
        if has_diffs:
            # If we already have an empty diff (from nested changes), update it
            if self in modified_entities:
                modified_entities[self].field_diffs.update(field_diffs)
            else:
                modified_entities[self] = EntityDiff.from_diff_dict(field_diffs)
        
        return bool(modified_entities), modified_entities

    def compute_diff(self, other: "Entity") -> EntityDiff:
        """Compute detailed differences between this entity and another entity."""
        _, field_diffs = compare_entity_fields(self, other)
        return EntityDiff.from_diff_dict(field_diffs)

    def entity_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Skip versioning fields, plus recursively dump nested Entities.
        """
        exclude_keys = set(kwargs.get('exclude', set()))
        exclude_keys |= {'id','created_at','parent_id','live_id','old_ids','lineage_id'}
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
        from __main__ import EntityRegistry
        ent = EntityRegistry.get(entity_id, expected_type=cls)
        return cast(Optional["Entity"], ent)

    @classmethod
    def list_all(cls: Type["Entity"]) -> List["Entity"]:
        """List all entities of this type."""
        from __main__ import EntityRegistry
        return EntityRegistry.list_by_type(cls)

    @classmethod
    def get_many(cls: Type["Entity"], ids: List[UUID]) -> List["Entity"]:
        """Get multiple entities by ID."""
        from __main__ import EntityRegistry
        return EntityRegistry.get_many(ids, expected_type=cls)

    def get_sub_entities(self) -> Set['Entity']:
        """Get all sub-entities of this entity."""
        nested: Set['Entity'] = set()
        for field_name, field_info in self.model_fields.items():
            value = getattr(self, field_name)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, Entity):
                        nested.add(item)
                        nested.update(item.get_sub_entities())
            elif isinstance(value, Entity):
                nested.add(value)
                nested.update(value.get_sub_entities())
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


##############################
# 6) SQL Storage Integration
##############################


try:
    from sqlmodel import SQLModel, Field, select, Session, create_engine
    from sqlalchemy import Column, JSON, inspect, or_
    from sqlalchemy.orm import joinedload, RelationshipProperty
    import importlib
    from datetime import timezone
    from typing import Tuple, Dict, List, Any, Optional, Type, ClassVar, Set, Union, cast
    
    def dynamic_import(path_str: str) -> Type[Entity]:
        """Import a class dynamically by its dotted path."""
        try:
            mod_name, cls_name = path_str.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            return getattr(mod, cls_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import {path_str}: {e}")

    class BaseEntitySQL(SQLModel, table=True):
        """Fallback table for storing any Entity if no specialized table is found."""
        # Database primary key (integer, auto-incremented)
        id: Optional[int] = Field(default=None, primary_key=True)
        
        # Entity versioning fields
        ecs_id: UUID = Field(default_factory=uuid4, index=True)
        lineage_id: UUID = Field(default_factory=uuid4, index=True)
        parent_id: Optional[UUID] = Field(default=None)  # Can't use foreign key to itself safely
        created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
        old_ids: List[UUID] = Field(default_factory=list, sa_column=Column(JSON))

        # The dotted Python path for the real class
        class_name: str = Field(...)

        # Non-versioning fields are stored here as JSON
        data: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

        def to_entity(self) -> Entity:
            cls_obj = dynamic_import(self.class_name)
            # Merge versioning fields + data
            combined = {
                "ecs_id": self.ecs_id,  # Changed from "id" to "ecs_id"
                "lineage_id": self.lineage_id,
                "parent_id": self.parent_id,
                "created_at": self.created_at,
                "old_ids": self.old_ids,
                "from_storage": True,
                **self.data
            }
            return cls_obj(**combined)

        @classmethod
        def from_entity(cls, entity: Entity) -> 'BaseEntitySQL':
            versioning_fields = {"ecs_id", "lineage_id", "parent_id", "created_at", "old_ids", "live_id", "from_storage", "force_parent_fork"}
            raw = entity.model_dump()
            data_only = {k: v for k, v in raw.items() if k not in versioning_fields}
            return cls(
                ecs_id=entity.ecs_id,
                lineage_id=entity.lineage_id,
                parent_id=entity.parent_id,
                created_at=entity.created_at,
                old_ids=entity.old_ids,
                class_name=f"{entity.__class__.__module__}.{entity.__class__.__qualname__}",
                data=data_only
            )
            
        def handle_relationships(self, entity: Entity, session: Session, orm_objects: Dict[UUID, Any]) -> None:
            """Generic relationship handler - can be implemented by each ORM model"""
            # Base implementation does nothing - subclasses can override
            pass

    class SqlEntityStorage(EntityStorage):
        """
        Optimized SQL-based storage implementation with proper relationship handling.
        
        Features:
        - Session reuse for related operations
        - Type-agnostic many-to-many relationship handling
        - Two-phase entity storage (store entities first, then establish relationships)
        - Comprehensive logging for debugging
        - Proper lookup by ecs_id instead of database id
        """
        def __init__(
            self,
            session_factory: Callable[..., Session],
            entity_to_orm_map: Dict[Type[Entity], Type[Any]]
        ) -> None:
            self._logger = logging.getLogger("SqlEntityStorage")
            self._session_factory = session_factory
            self._entity_to_orm_map = entity_to_orm_map
            self._inference_orchestrator: Optional[object] = None
            
            # If BaseEntitySQL is available, use it as fallback
            if BaseEntitySQL and Entity not in self._entity_to_orm_map:
                self._entity_to_orm_map[Entity] = BaseEntitySQL
                self._logger.info("Added BaseEntitySQL as fallback ORM mapping")
            
            # Cache to avoid repeated lookups
            self._entity_class_map: Dict[UUID, Type[Entity]] = {}
            self._logger.info(f"Initialized SQL storage with {len(entity_to_orm_map)} entity mappings")
        
        def get_session(self, existing_session: Optional[Session] = None) -> Tuple[Session, bool]:
            """
            Get a session - either the provided one or a new one.
            
            Args:
                existing_session: Optional existing session to reuse
                
            Returns:
                Tuple of (session, should_close_when_done)
            """
            if existing_session is not None:
                self._logger.debug("Reusing provided session")
                return existing_session, False
                
            self._logger.debug("Creating new session")
            return self._session_factory(), True
        
        def has_entity(self, entity_id: UUID, session: Optional[Session] = None) -> bool:
            """
            Check if an entity exists in storage.
            
            Args:
                entity_id: UUID of the entity to check
                session: Optional session to reuse
                
            Returns:
                True if the entity exists, False otherwise
            """
            # Check cache first
            if entity_id in self._entity_class_map:
                self._logger.debug(f"Entity {entity_id} found in cache")
                return True
            
            # Query database
            session_obj, should_close = self.get_session(session)
            try:
                for entity_cls, orm_cls in self._entity_to_orm_map.items():
                    stmt = select(orm_cls).where(orm_cls.ecs_id == entity_id)
                    result = session_obj.exec(stmt).first()
                    if result is not None:
                        entity = result.to_entity()
                        self._entity_class_map[entity_id] = type(entity)
                        self._logger.debug(f"Entity {entity_id} found in database")
                        return True
                return False
            except Exception as e:
                self._logger.error(f"Error checking entity existence: {str(e)}")
                return False
            finally:
                if should_close:
                    session_obj.close()
        
        def register(self, entity_or_id: Union[Entity, UUID], session: Optional[Session] = None) -> Optional[Entity]:
            """Register a root entity and all its sub-entities in a single transaction."""
            if isinstance(entity_or_id, UUID):
                return self.get(entity_or_id, None, session=session)
                
            entity = entity_or_id
            self._logger.info(f"Registering root entity {type(entity).__name__}({entity.ecs_id})")
            
            # Create a new session if we don't have one
            own_session = session is None
            session_obj = session or self._session_factory()
            
            try:
                # Store the entire entity tree
                self._store_entity_tree(entity, session_obj)
                
                # Commit if we created the session
                if own_session:
                    session_obj.commit()
                    
                self._logger.info(f"Successfully registered entity tree for {type(entity).__name__}({entity.ecs_id})")
                return entity
                    
            except Exception as e:
                if own_session:
                    self._logger.error(f"Error registering entity tree, rolling back: {str(e)}")
                    session_obj.rollback()
                else:
                    self._logger.error(f"Error registering entity tree (using external session): {str(e)}")
                return None
            finally:
                if own_session:
                    session_obj.close()
        
        def get_cold_snapshot(self, entity_id: UUID, session: Optional[Session] = None) -> Optional[Entity]:
            """
            Get the stored version of an entity.
            
            Args:
                entity_id: UUID of the entity to retrieve
                session: Optional session to reuse
                
            Returns:
                The stored entity, or None if not found
            """
            self._logger.debug(f"Getting cold snapshot for entity {entity_id}")
            
            # Try with known class first if cached
            cls_maybe = self._entity_class_map.get(entity_id)
            self._logger.debug(f"Lookup cached type: {cls_maybe.__name__ if cls_maybe else 'None'}")
            
            session_obj, should_close = self.get_session(session)
            try:
                if cls_maybe:
                    orm_cls = self._entity_to_orm_map.get(cls_maybe)
                    if orm_cls:
                        # Use select with where clause instead of session.get()
                        stmt = select(orm_cls).where(orm_cls.ecs_id == entity_id)
                        # Add joinedload for each relationship using SQLAlchemy's inspect
                        for rel_name, rel_prop in self._get_relationship_properties(orm_cls).items():
                            attr = getattr(orm_cls, rel_name)
                            stmt = stmt.options(joinedload(attr))
                            
                        result = session_obj.exec(stmt).first()
                        if result:
                            self._logger.debug(f"Found entity {entity_id} in {orm_cls.__name__} table")
                            entity = result.to_entity()
                            return entity
                
                # Fallback to scanning all tables by ecs_id
                self._logger.debug(f"Scanning all tables for entity {entity_id}")
                for entity_cls, orm_cls in self._entity_to_orm_map.items():
                    # Use select with where clause and joinedload for relationships
                    stmt = select(orm_cls).where(orm_cls.ecs_id == entity_id)
                    for rel_name, rel_prop in self._get_relationship_properties(orm_cls).items():
                        attr = getattr(orm_cls, rel_name)
                        stmt = stmt.options(joinedload(attr))
                        
                    result = session_obj.exec(stmt).first()
                    if result:
                        entity = result.to_entity()
                        self._entity_class_map[entity_id] = type(entity)
                        self._logger.info(f"Found entity {entity_id} in {orm_cls.__name__} table")
                        return entity
                
                self._logger.warning(f"Entity {entity_id} not found in any table")
                return None
                
            except Exception as e:
                self._logger.error(f"Error getting cold snapshot: {str(e)}")
                return None
            finally:
                if should_close:
                    self._logger.debug("Closing session after get_cold_snapshot")
                    session_obj.close()
        
        def get(self, entity_id: UUID, expected_type: Optional[Type[Entity]] = None, 
                session: Optional[Session] = None) -> Optional[Entity]:
            """
            Get an entity by ID with optional type checking.
            
            Args:
                entity_id: UUID of the entity to retrieve
                expected_type: Optional type to check against
                session: Optional session to reuse
                
            Returns:
                The entity, or None if not found or type mismatch
            """
            self._logger.debug(f"Getting entity {entity_id}" + 
                            (f" (expected type: {expected_type.__name__})" if expected_type else ""))
            
            try:
                entity = self.get_cold_snapshot(entity_id, session=session)
                if not entity:
                    self._logger.debug(f"Entity {entity_id} not found")
                    return None
                
                if expected_type and not isinstance(entity, expected_type):
                    self._logger.error(f"Type mismatch: got {type(entity).__name__}, expected {expected_type.__name__}")
                    return None
                
                # Create a warm copy
                warm_copy = deepcopy(entity)
                warm_copy.live_id = uuid4()
                warm_copy.from_storage = True
                
                self._logger.debug(f"Returning warm copy of entity {entity_id} (live_id: {warm_copy.live_id})")
                return warm_copy
                
            except Exception as e:
                self._logger.error(f"Error getting entity: {str(e)}")
                return None
        
        def list_by_type(self, entity_type: Type[Entity], session: Optional[Session] = None) -> List[Entity]:
            """
            List all entities of a specific type.
            
            Args:
                entity_type: Type of entities to list
                session: Optional session to reuse
                
            Returns:
                List of entities of the specified type
            """
            self._logger.debug(f"Listing entities of type {entity_type.__name__}")
            
            orm_cls = self._entity_to_orm_map.get(entity_type)
            if not orm_cls:
                # Try fallback to BaseEntitySQL if entity_type is Entity
                if entity_type is Entity and Entity in self._entity_to_orm_map:
                    orm_cls = self._entity_to_orm_map[Entity]
                else:
                    self._logger.warning(f"No ORM mapping found for {entity_type.__name__}")
                    return []
            
            session_obj, should_close = self.get_session(session)
            try:
                # Use select with joinedload for relationships
                stmt = select(orm_cls)
                for rel_name, rel_prop in self._get_relationship_properties(orm_cls).items():
                    attr = getattr(orm_cls, rel_name)
                    stmt = stmt.options(joinedload(attr))
                    
                entities: List[Entity] = []
                results = session_obj.exec(stmt).all()
                for result in results:
                    entity = result.to_entity()
                    self._entity_class_map[entity.ecs_id] = type(entity)
                    entities.append(entity)
                    
                self._logger.debug(f"Found {len(entities)} entities of type {entity_type.__name__}")
                return entities
                
            except Exception as e:
                self._logger.error(f"Error listing entities by type: {str(e)}")
                return []
            finally:
                if should_close:
                    self._logger.debug("Closing session after list_by_type")
                    session_obj.close()
        
        def get_many(self, entity_ids: List[UUID], expected_type: Optional[Type[Entity]] = None,
                    session: Optional[Session] = None) -> List[Entity]:
            """
            Get multiple entities by ID.
            
            Args:
                entity_ids: List of UUIDs to retrieve
                expected_type: Optional type to check against
                session: Optional session to reuse
                
            Returns:
                List of entities matching the IDs and type
            """
            self._logger.debug(f"Getting {len(entity_ids)} entities" +
                            (f" (expected type: {expected_type.__name__})" if expected_type else ""))
            
            session_obj, should_close = self.get_session(session)
            try:
                results: List[Entity] = []
                
                if expected_type:
                    # If we know the type, we can do a single query
                    orm_cls = self._entity_to_orm_map.get(expected_type)
                    if orm_cls:
                        # Use OR conditions for ecs_id filtering
                        conditions = [orm_cls.ecs_id == entity_id for entity_id in entity_ids]
                        if conditions:
                            stmt = select(orm_cls).where(or_(*conditions))
                            for rel_name, rel_prop in self._get_relationship_properties(orm_cls).items():
                                attr = getattr(orm_cls, rel_name)
                                stmt = stmt.options(joinedload(attr))
                                
                            db_results = session_obj.exec(stmt).all()
                            for result in db_results:
                                entity = result.to_entity()
                                self._entity_class_map[entity.ecs_id] = type(entity)
                                
                                # Create warm copy
                                warm_copy = deepcopy(entity)
                                warm_copy.live_id = uuid4()
                                warm_copy.from_storage = True
                                
                                results.append(warm_copy)
                else:
                    # Otherwise, query each ID individually (less efficient)
                    for entity_id in entity_ids:
                        entity = self.get(entity_id, expected_type, session=session_obj)
                        if entity:
                            results.append(entity)
                
                self._logger.debug(f"Retrieved {len(results)}/{len(entity_ids)} entities")
                return results
                
            except Exception as e:
                self._logger.error(f"Error getting multiple entities: {str(e)}")
                return []
            finally:
                if should_close:
                    self._logger.debug("Closing session after get_many")
                    session_obj.close()
        
        def get_registry_status(self) -> Dict[str, Any]:
            """Get status information about the registry."""
            return {
                "storage": "sql",
                "known_ids_in_cache": len(self._entity_class_map)
            }
        
        def set_inference_orchestrator(self, orchestrator: object) -> None:
            """Set an inference orchestrator."""
            self._inference_orchestrator = orchestrator
        
        def get_inference_orchestrator(self) -> Optional[object]:
            """Get the current inference orchestrator."""
            return self._inference_orchestrator
        
        def clear(self) -> None:
            """Clear cached data (doesn't affect database)."""
            self._entity_class_map.clear()
            self._logger.warning("SqlEntityStorage.clear() only clears caches, not database data")
        
        def get_lineage_entities(self, lineage_id: UUID, session: Optional[Session] = None) -> List[Entity]:
            """
            Get all entities with a specific lineage ID.
            
            Args:
                lineage_id: Lineage ID to query
                session: Optional session to reuse
                
            Returns:
                List of entities with the specified lineage ID
            """
            self._logger.debug(f"Getting entities with lineage ID {lineage_id}")
            
            session_obj, should_close = self.get_session(session)
            try:
                entities: List[Entity] = []
                
                for entity_cls, orm_cls in self._entity_to_orm_map.items():
                    # Use select with where clause and joinedload for relationships
                    stmt = select(orm_cls).where(orm_cls.lineage_id == lineage_id)
                    for rel_name, rel_prop in self._get_relationship_properties(orm_cls).items():
                        attr = getattr(orm_cls, rel_name)
                        stmt = stmt.options(joinedload(attr))
                        
                    results = session_obj.exec(stmt).unique().all()
                    
                    for result in results:
                        entity = result.to_entity()
                        self._entity_class_map[entity.ecs_id] = type(entity)
                        entities.append(entity)
                
                self._logger.debug(f"Found {len(entities)} entities with lineage ID {lineage_id}")
                return entities
                
            except Exception as e:
                self._logger.error(f"Error getting lineage entities: {str(e)}")
                return []
            finally:
                if should_close:
                    self._logger.debug("Closing session after get_lineage_entities")
                    session_obj.close()
        
        def has_lineage_id(self, lineage_id: UUID, session: Optional[Session] = None) -> bool:
            """
            Check if a lineage ID exists.
            
            Args:
                lineage_id: Lineage ID to check
                session: Optional session to reuse
                
            Returns:
                True if the lineage ID exists, False otherwise
            """
            entities = self.get_lineage_entities(lineage_id, session=session)
            return len(entities) > 0
        
        def get_lineage_ids(self, lineage_id: UUID, session: Optional[Session] = None) -> List[UUID]:
            """
            Get all entity IDs with a specific lineage ID.
            
            Args:
                lineage_id: Lineage ID to query
                session: Optional session to reuse
                
            Returns:
                List of entity IDs with the specified lineage ID
            """
            return [entity.ecs_id for entity in self.get_lineage_entities(lineage_id, session=session)]
        
        def _get_orm_class(self, entity: Entity) -> Optional[Type[Any]]:
            """
            Get the appropriate ORM class for an entity.
            
            Args:
                entity: Entity to get ORM class for
                
            Returns:
                ORM class for the entity, or None if not found
            """
            try:
                # Try exact class match
                orm_cls = self._entity_to_orm_map.get(type(entity))
                if orm_cls:
                    self._logger.debug(f"Found exact ORM match for {type(entity).__name__}: {orm_cls.__name__}")
                    return orm_cls
                
                # Try parent classes
                for entity_cls, orm_cls in self._entity_to_orm_map.items():
                    if isinstance(entity, entity_cls):
                        self._logger.debug(f"Found parent ORM match for {type(entity).__name__}: {orm_cls.__name__}")
                        return orm_cls
                    
                self._logger.warning(f"No ORM mapping found for {type(entity).__name__}")
                return None
                
            except Exception as e:
                self._logger.error(f"Error getting ORM class: {str(e)}")
                return None

        def _get_relationship_properties(self, orm_cls: Type[Any]) -> Dict[str, RelationshipProperty]:
            """
            Get relationship properties for an ORM class using SQLAlchemy's inspect.
            
            Args:
                orm_cls: ORM class to get relationships for
                
            Returns:
                Dictionary of relationship name to RelationshipProperty
            """
            try:
                inspector = inspect(orm_cls)
                relationships = {}
                for rel_name, rel_prop in inspector.relationships.items():
                    relationships[rel_name] = rel_prop
                return relationships
            except Exception as e:
                self._logger.error(f"Error getting relationship properties for {orm_cls.__name__}: {str(e)}")
                return {}

        def _store_entity_tree(self, entity: Entity, session: Session) -> None:
            """
            Store an entire entity tree in a single database transaction.
            
            Args:
                entity: Root entity of the tree
                session: Session to use
            """
            self._logger.debug(f"Storing entity tree for {type(entity).__name__}({entity.ecs_id})")
            
            # Collect all entities in the tree
            entities_to_store: Dict[UUID, Entity] = {}
            orm_objects: Dict[UUID, Any] = {}
            
            def collect_entities(e: Entity) -> None:
                if e.ecs_id not in entities_to_store:
                    # Create snapshot for storage
                    snapshot = create_cold_snapshot(e)
                    entities_to_store[e.ecs_id] = snapshot
                    # Collect nested entities
                    for sub in e.get_sub_entities():
                        collect_entities(sub)
            
            # Collect all entities in the tree
            collect_entities(entity)
            self._logger.info(f"Collected {len(entities_to_store)} entities to store")
            
            # Store all entities first
            for e in entities_to_store.values():
                # Get ORM class for this entity type
                orm_cls = self._get_orm_class(e)
                if not orm_cls:
                    self._logger.error(f"No ORM mapping found for {type(e)}")
                    continue
                
                # Check if entity exists in database
                stmt = select(orm_cls).where(orm_cls.ecs_id == e.ecs_id)
                existing = session.exec(stmt).first()
                
                # Convert to ORM object
                if existing:
                    # Update existing record
                    self._logger.debug(f"Updating existing database record for {type(e).__name__}({e.ecs_id})")
                    try:
                        # Get fields excluding ID
                        fields = e.model_dump(exclude={"id"})
                        for field, value in fields.items():
                            if hasattr(existing, field):
                                setattr(existing, field, value)
                    except AttributeError:
                        # Fall back to dict for older Pydantic versions
                        fields = {field: getattr(e, field) for field in e.__dict__ 
                                 if field != "id" and not field.startswith("_")}
                        for field, value in fields.items():
                            setattr(existing, field, value)
                        
                    orm_objects[e.ecs_id] = existing
                else:
                    # Add new record
                    self._logger.debug(f"Creating new database record for {type(e).__name__}({e.ecs_id})")
                    orm_obj = orm_cls.from_entity(e)
                    session.add(orm_obj)
                    orm_objects[e.ecs_id] = orm_obj
            
            # Flush to get IDs assigned
            session.flush()
            
            # Now handle relationships
            self._logger.debug(f"Handling relationships for {len(entities_to_store)} entities")
            for ent_id, orm_obj in orm_objects.items():
                entity = entities_to_store[ent_id]
                
                # Use handle_relationships method if available
                if hasattr(orm_obj, "handle_relationships") and callable(getattr(orm_obj, "handle_relationships")):
                    try:
                        handler = getattr(orm_obj, "handle_relationships")
                        handler(entity, session, orm_objects)
                    except Exception as e:
                        self._logger.error(f"Error in handle_relationships for {type(entity).__name__}: {str(e)}")
                    
            # Refresh all objects to ensure all changes are loaded
            for orm_obj in orm_objects.values():
                session.refresh(orm_obj)
            
            # Update cache
            for e in entities_to_store.values():
                self._entity_class_map[e.ecs_id] = type(e)
            
            self._logger.info(f"Successfully stored entity tree with {len(entities_to_store)} entities")

except ImportError:
    # SQLModel not available, skip SQL storage implementation
    BaseEntitySQL = None  # type: ignore
    SqlEntityStorage = None  # type: ignore


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

    @classmethod
    def get_lineage_tree_sorted(cls, lineage_id: UUID) -> Dict[str, Any]:
        """Get a lineage tree with entities sorted by creation time."""
        tree = cls.build_lineage_tree(lineage_id)
        if not tree:
            return {
                "nodes": {},
                "edges": [],
                "root": None,
                "sorted_ids": [],
                "diffs": {}
            }
            
        # Sort nodes by creation time
        sorted_items = sorted(tree.items(), key=lambda x: x[1]["created_at"])
        sorted_ids = [item[0] for item in sorted_items]
        
        # Extract edges and diffs
        edges = []
        diffs = {}
        for node_id, node_data in tree.items():
            parent_id = node_data["parent_id"]
            if parent_id:
                edges.append((parent_id, node_id))
                if node_data["diff_from_parent"]:
                    diffs[node_id] = node_data["diff_from_parent"]
                    
        # Find root node
        root_candidates = [node_id for node_id, data in tree.items() if not data["parent_id"]]
        root_id = root_candidates[0] if root_candidates else None
        
        return {
            "nodes": tree,
            "edges": edges,
            "root": root_id,
            "sorted_ids": sorted_ids,
            "diffs": diffs
        }

    @classmethod
    def get_lineage_mermaid(cls, lineage_id: UUID) -> str:
        """Generate a Mermaid diagram for a lineage tree."""
        data = cls.get_lineage_tree_sorted(lineage_id)
        if not data["nodes"]:
            return "```mermaid\ngraph TD\n  No data available\n```"

        lines = ["```mermaid", "graph TD"]

        # Helper for formatting values in diagram
        def format_value(val: Any) -> str:
            s = str(val)
            return s[:15] + "..." if len(s) > 15 else s

        # Add nodes to diagram
        for node_id, node_data in data["nodes"].items():
            entity = node_data["entity"]
            class_name = type(entity).__name__
            short_id = str(node_id)[:8]
            
            if not node_data["parent_id"]:
                # Root node with summary
                data_items = list(node_data["data"].items())[:3]
                summary = "\\n".join(f"{k}={format_value(v)}" for k, v in data_items)
                lines.append(f"  {node_id}[\"{class_name}\\n{short_id}\\n{summary}\"]")
            else:
                # Child node with change count
                diff = data["diffs"].get(node_id, {})
                change_count = len(diff)
                lines.append(f"  {node_id}[\"{class_name}\\n{short_id}\\n({change_count} changes)\"]")

        # Add edges to diagram
        for parent_id, child_id in data["edges"]:
            diff = data["diffs"].get(child_id, {})
            if diff:
                # Edge with diff labels
                label_parts = []
                for field, info in diff.items():
                    diff_type = info.get("type")
                    if diff_type == "modified":
                        label_parts.append(f"{field} mod")
                    elif diff_type == "added":
                        label_parts.append(f"+{field}")
                    elif diff_type == "removed":
                        label_parts.append(f"-{field}")
                
                # Truncate if too many changes
                if len(label_parts) > 3:
                    label_parts = label_parts[:3] + [f"...({len(diff) - 3} more)"]
                    
                label = "\\n".join(label_parts)
                lines.append(f"  {parent_id} -->|\"{label}\"| {child_id}")
            else:
                # Simple edge
                lines.append(f"  {parent_id} --> {child_id}")

        lines.append("```")
        return "\n".join(lines)


##############################
# 8) Entity Tracing
##############################

def _find_entities(obj: Any) -> Dict[int, Entity]:
    """
    Recursively scan an object for Entity instances.
    Returns a dictionary mapping object IDs to entities.
    """
    found: Dict[int, Entity] = {}
    
    def scan(item: Any) -> None:
        if isinstance(item, Entity):
            found[id(item)] = item
        elif isinstance(item, (list, tuple, set)):
            for element in item:
                scan(element)
        elif isinstance(item, dict):
            for value in item.values():
                scan(value)
    
    scan(obj)
    return found

def _check_and_process_entities(entities: Dict[int, Entity], fork_if_modified: bool = True) -> None:
    """
    Check entities for modifications and optionally fork them.
    Process in bottom-up order (nested entities first).
    """
    from __main__ import EntityRegistry
    
    # Build dependency graph
    dependency_graph: Dict[int, List[int]] = {id(e): [] for e in entities.values()}
    for entity_id, entity in entities.items():
        # Find all nested entities
        for sub in entity.get_sub_entities():
            nested_id = id(sub)
            if nested_id in entities:
                # Add dependency: entity depends on nested_entity
                dependency_graph[entity_id].append(nested_id)
    
    # Topological sort (process leaves first)
    processed: Set[int] = set()
    
    def process_entity(entity_id: int) -> None:
        if entity_id in processed:
            return
            
        # Process dependencies first
        for dep_id in dependency_graph[entity_id]:
            if dep_id not in processed:
                process_entity(dep_id)
                
        # Process this entity
        entity = entities[entity_id]
        cold = EntityRegistry.get_cold_snapshot(entity.ecs_id)
        
        if cold and entity.has_modifications(cold) and fork_if_modified:
            entity.fork()
            
        processed.add(entity_id)
    
    # Process all entities
    for entity_id in entities:
        if entity_id not in processed:
            process_entity(entity_id)


def entity_tracer(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Improved decorator that checks and forks entities before and after function call.
    Handles async functions and detects entities in a bottom-up order.
    """
    # Check if function is async by looking for _is_coroutine attribute
    is_async = getattr(func, '_is_coroutine', False) or hasattr(func, '__await__')
    
    if is_async:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Find entities before function call
            entities_before = _find_entities((args, kwargs))
            
            # Check and process entities before function call
            _check_and_process_entities(entities_before)
            
            # Call the function
            result = await func(*args, **kwargs)
            
            # Check and process entities after function call
            _check_and_process_entities(entities_before)
            
            # Handle result if it's an entity
            if isinstance(result, Entity) and id(result) in entities_before:
                return entities_before[id(result)]
                
            return result
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Find entities before function call
            entities_before = _find_entities((args, kwargs))
            
            # Check and process entities before function call
            _check_and_process_entities(entities_before)
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Check and process entities after function call
            _check_and_process_entities(entities_before)
            
            # Handle result if it's an entity
            if isinstance(result, Entity) and id(result) in entities_before:
                return entities_before[id(result)]
                
            return result
        return sync_wrapper