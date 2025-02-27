############################################################
# entity.py (refactored)
############################################################

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
E = TypeVar('E', bound='Entity')
SQMT = TypeVar('SQMT', bound='SQLModelType')

# Type for SQL model classes that implement to_entity/from_entity
class SQLModelType(Protocol):
    @classmethod
    def from_entity(cls, entity: 'Entity') -> 'SQLModelType': ...
    def to_entity(self) -> 'Entity': ...
    id: UUID
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

def get_nested_entities(entity: 'Entity') -> Dict[str, List['Entity']]:
    """
    Get all nested entities within an entity.
    Returns a dictionary mapping field names to lists of entities.
    """
    nested: Dict[str, List['Entity']] = {}
    
    for field_name, field_value in entity.model_dump().items():
        if isinstance(field_value, Entity):
            nested[field_name] = [field_value]
        elif isinstance(field_value, list):
            entities = [item for item in field_value if isinstance(item, Entity)]
            if entities:
                nested[field_name] = entities
    
    return nested

##############################
# 4) The Entity + Diff
##############################

@runtime_checkable
class HasID(Protocol):
    """Protocol requiring an `id: UUID` field."""
    id: UUID

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

class Entity(BaseModel):
    """
    Base class for registry-integrated, serializable entities.
    Supports versioning, nested entity tracking, and registry integration.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    live_id: UUID = Field(default_factory=uuid4, description="Live/warm identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    parent_id: Optional[UUID] = None
    lineage_id: UUID = Field(default_factory=uuid4)
    old_ids: List[UUID] = Field(default_factory=list)
    from_storage: bool = Field(default=False, description="Whether the entity was loaded from storage")
    
    model_config = {
        "json_encoders": {
            UUID: str,
            datetime: lambda dt: dt.isoformat()
        }
    }

    def register_entity(self: "Entity") -> "Entity":
        """Register this entity with the registry."""
        from __main__ import EntityRegistry
        
        # Register nested entities first (bottom-up approach)
        self._register_nested_entities()
        
        # Check if entity exists and if it has changes
        if not EntityRegistry.has_entity(self.id):
            EntityRegistry.register(self)
        elif not self.from_storage:
            cold = EntityRegistry.get_cold_snapshot(self.id)
            if cold and self.has_modifications(cold):
                self.fork()
                
        return self

    def _register_nested_entities(self) -> None:
        """Register all nested entities first to ensure bottom-up processing."""
        from __main__ import EntityRegistry
        
        # Get all nested entities
        nested_entities = get_nested_entities(self)
        
        # Register each entity
        for field_name, entities in nested_entities.items():
            for entity in entities:
                if isinstance(entity, Entity):
                    # Check if entity has changes and fork if needed
                    if not EntityRegistry.has_entity(entity.id):
                        entity.register_entity()
                    else:
                        cold = EntityRegistry.get_cold_snapshot(entity.id)
                        if not entity.from_storage and cold and entity.has_modifications(cold):
                            # Fork returns the new entity
                            new_entity = entity.fork()
                            
                            # Update the reference in the parent
                            if isinstance(getattr(self, field_name), list):
                                # For list fields, find and replace the entity
                                entity_list = getattr(self, field_name)
                                for i, e in enumerate(entity_list):
                                    if isinstance(e, Entity) and e.id == entity.id:
                                        entity_list[i] = new_entity
                            else:
                                # For single reference fields
                                setattr(self, field_name, new_entity)

    @model_validator(mode='after')
    def _auto_register(self) -> Self:
        """Automatically register the entity when validated."""
        if not self.from_storage:  # Only auto-register if not from storage
            entity = self.register_entity()
        return self

    def fork(self, force: bool = False, **kwargs: Any) -> "Entity":
        """
        Create a new version of this entity with a new ID.
        
        Args:
            force: Force forking even if no changes detected
            **kwargs: Fields to modify in the new fork
            
        Returns:
            The new forked entity
        """
        from __main__ import EntityRegistry
        
        # Fork nested entities first (bottom-up approach)
        nested_changes = self._fork_nested_entities()
        
        # Get cold snapshot
        cold = EntityRegistry.get_cold_snapshot(self.id)
        if cold is None:
            EntityRegistry.register(self)
            return self

        # Check for direct field changes
        changed = bool(kwargs)
        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)

        # Determine if forking is needed
        needs_fork = (
            force or 
            changed or 
            nested_changes or 
            (cold and self.has_modifications(cold))
        )
        
        if not needs_fork:
            return self

        # Create new entity version
        old_id = self.id
        self.id = uuid4()
        self.parent_id = old_id
        self.old_ids.append(old_id)
        
        # Register new version
        EntityRegistry.register(self)
        return self

    def _fork_nested_entities(self) -> bool:
        """
        Fork all nested entities that have modifications.
        Returns True if any nested entities were forked.
        """
        from __main__ import EntityRegistry
        
        any_changes = False
        nested_entities = get_nested_entities(self)
        
        for field_name, entities in nested_entities.items():
            for i, entity in enumerate(entities):
                if isinstance(entity, Entity):
                    # Check if entity has changes
                    cold = EntityRegistry.get_cold_snapshot(entity.id)
                    if cold and entity.has_modifications(cold):
                        # Fork and update reference
                        new_entity = entity.fork()
                        any_changes = True
                        
                        # Update the reference in the parent
                        if isinstance(getattr(self, field_name), list):
                            # For list fields, find and replace the entity
                            entity_list = getattr(self, field_name)
                            for j, e in enumerate(entity_list):
                                if isinstance(e, Entity) and e.id == entity.id:
                                    entity_list[j] = new_entity
                        else:
                            # For single reference fields
                            setattr(self, field_name, new_entity)
        
        return any_changes

    def has_modifications(self, other: "Entity") -> bool:
        """Check if this entity differs from another entity."""
        has_diffs, _ = compare_entity_fields(self, other)
        return has_diffs

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
    Improved in-memory storage with optimized operations.
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

        e = entity_or_id
        old = self._registry.get(e.id)
        
        if not old:
            self._store_cold_snapshot(e)
            return e
        
        # Check for modifications
        if e.has_modifications(old):
            e.fork(force=True)
            
        return e

    def get(self, entity_id: UUID, expected_type: Optional[Type[Entity]]) -> Optional[Entity]:
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

    def get_many(self, entity_ids: List[UUID], expected_type: Optional[Type[Entity]]) -> List[Entity]:
        """Get multiple entities by ID."""
        out: List[Entity] = []
        for eid in entity_ids:
            g = self.get(eid, expected_type)
            if g is not None:
                out.append(g)
        return out

    def get_registry_status(self) -> Dict[str, Any]:
        """Get status information about the registry."""
        return {
            "in_memory": True,
            "entity_count": len(self._registry),
            "lineage_count": len(self._lineages),
        }

    def set_inference_orchestrator(self, orchestrator: object) -> None:
        """Set an inference orchestrator for this storage."""
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
        return [e.id for e in self._registry.values() if e.lineage_id == lineage_id]

    def _store_cold_snapshot(self, e: Entity) -> None:
        """Store a cold snapshot of an entity."""
        snap = create_cold_snapshot(e)
        self._registry[e.id] = snap

        # Update lineage tracking
        if e.lineage_id not in self._lineages:
            self._lineages[e.lineage_id] = []
        if e.id not in self._lineages[e.lineage_id]:
            self._lineages[e.lineage_id].append(e.id)


##############################
# 6) SQL Storage Integration
##############################

try:
    from sqlmodel import SQLModel, Field, select, Session, create_engine
    from sqlalchemy import Column, JSON
    import importlib
    from datetime import timezone
    
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
        id: UUID = Field(default_factory=uuid4, primary_key=True)
        lineage_id: UUID = Field(default_factory=uuid4, index=True)
        parent_id: Optional[UUID] = Field(default=None)
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
                "id": self.id,
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
            versioning_fields = {"id", "lineage_id", "parent_id", "created_at", "old_ids", "live_id"}
            raw = entity.model_dump()
            data_only = {k: v for k, v in raw.items() if k not in versioning_fields}
            return cls(
                id=entity.id,
                lineage_id=entity.lineage_id,
                parent_id=entity.parent_id,
                created_at=entity.created_at,
                old_ids=entity.old_ids,
                class_name=f"{entity.__class__.__module__}.{entity.__class__.__qualname__}",
                data=data_only
            )
    
    class SqlEntityStorage(EntityStorage):
        """
        Optimized SQL-based storage implementation.
        Uses SQLModel patterns consistently.
        """
        def __init__(
            self,
            session_factory: Callable[..., Session],
            entity_to_orm_map: Dict[Type[Entity], Type[SQLModelType]]
        ) -> None:
            self._logger = logging.getLogger("SqlEntityStorage")
            self._session_factory = session_factory
            self._entity_to_orm_map = entity_to_orm_map
            self._inference_orchestrator: Optional[object] = None
            
            # If BaseEntitySQL is available, use it as fallback
            if BaseEntitySQL and Entity not in self._entity_to_orm_map:
                self._entity_to_orm_map[Entity] = BaseEntitySQL
            
            # Cache to avoid repeated lookups
            self._entity_class_map: Dict[UUID, Type[Entity]] = {}
        
        def has_entity(self, entity_id: UUID) -> bool:
            """Check if an entity exists in storage."""
            # Check cache first
            if entity_id in self._entity_class_map:
                return True
                
            # Query database
            with self._session_factory() as session:
                for entity_cls, orm_cls in self._entity_to_orm_map.items():
                    stmt = select(orm_cls).where(orm_cls.id == entity_id)
                    result = session.exec(stmt).first()
                    if result is not None:
                        entity = result.to_entity()
                        self._entity_class_map[entity_id] = type(entity)
                        return True
            return False
        
        def get_cold_snapshot(self, entity_id: UUID) -> Optional[Entity]:
            """Get the stored version of an entity."""
            # Try with known class first if cached
            cls_maybe = self._entity_class_map.get(entity_id)
            
            with self._session_factory() as session:
                if cls_maybe:
                    orm_cls = self._entity_to_orm_map.get(cls_maybe)
                    if orm_cls:
                        stmt = select(orm_cls).where(orm_cls.id == entity_id)
                        result = session.exec(stmt).first()
                        if result:
                            return result.to_entity()
                
                # Fallback to scanning all tables
                for entity_cls, orm_cls in self._entity_to_orm_map.items():
                    stmt = select(orm_cls).where(orm_cls.id == entity_id)
                    result = session.exec(stmt).first()
                    if result:
                        entity = result.to_entity()
                        self._entity_class_map[entity_id] = type(entity)
                        return entity
            return None
        
        def register(self, entity_or_id: Union[Entity, UUID]) -> Optional[Entity]:
            """Register an entity or retrieve it by ID."""
            if isinstance(entity_or_id, UUID):
                return self.get(entity_or_id, None)
                
            entity = entity_or_id
            
            # Check for modifications
            old = self.get_cold_snapshot(entity.id)
            if old and entity.has_modifications(old):
                entity.fork(force=True)
                return self.register(entity)  # Re-register with new ID
                
            # Find appropriate ORM class
            orm_cls = self._get_orm_class(entity)
            if not orm_cls:
                self._logger.error(f"No ORM mapping found for {type(entity)}")
                return None
                
            # Convert to ORM model and save
            orm_obj = orm_cls.from_entity(entity)
            with self._session_factory() as session:
                session.add(orm_obj)
                session.commit()
                
            # Update cache
            self._entity_class_map[entity.id] = type(entity)
            return entity
        
        def get(self, entity_id: UUID, expected_type: Optional[Type[Entity]]) -> Optional[Entity]:
            """Get an entity by ID with optional type checking."""
            entity = self.get_cold_snapshot(entity_id)
            if not entity:
                return None
                
            if expected_type and not isinstance(entity, expected_type):
                self._logger.error(f"Type mismatch: got {type(entity).__name__}, expected {expected_type.__name__}")
                return None
                
            # Create a warm copy
            warm_copy = deepcopy(entity)
            warm_copy.live_id = uuid4()
            warm_copy.from_storage = True
            
            return warm_copy
        
        def list_by_type(self, entity_type: Type[Entity]) -> List[Entity]:
            """List all entities of a specific type."""
            orm_cls = self._entity_to_orm_map.get(entity_type)
            if not orm_cls:
                # Try fallback to BaseEntitySQL if entity_type is Entity
                if entity_type is Entity and Entity in self._entity_to_orm_map:
                    orm_cls = self._entity_to_orm_map[Entity]
                else:
                    return []
                    
            entities: List[Entity] = []
            with self._session_factory() as session:
                stmt = select(orm_cls)
                results = session.exec(stmt).all()
                for result in results:
                    entity = result.to_entity()
                    self._entity_class_map[entity.id] = type(entity)
                    entities.append(entity)
            return entities
        
        def get_many(self, entity_ids: List[UUID], expected_type: Optional[Type[Entity]]) -> List[Entity]:
            """Get multiple entities by ID."""
            results: List[Entity] = []
            for entity_id in entity_ids:
                entity = self.get(entity_id, expected_type)
                if entity:
                    results.append(entity)
            return results
        
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
        
        def get_lineage_entities(self, lineage_id: UUID) -> List[Entity]:
            """Get all entities with a specific lineage ID."""
            entities: List[Entity] = []
            with self._session_factory() as session:
                for entity_cls, orm_cls in self._entity_to_orm_map.items():
                    stmt = select(orm_cls).where(orm_cls.lineage_id == lineage_id)
                    results = session.exec(stmt).all()
                    for result in results:
                        entity = result.to_entity()
                        self._entity_class_map[entity.id] = type(entity)
                        entities.append(entity)
            return entities
        
        def has_lineage_id(self, lineage_id: UUID) -> bool:
            """Check if a lineage ID exists."""
            entities = self.get_lineage_entities(lineage_id)
            return len(entities) > 0
        
        def get_lineage_ids(self, lineage_id: UUID) -> List[UUID]:
            """Get all entity IDs with a specific lineage ID."""
            return [entity.id for entity in self.get_lineage_entities(lineage_id)]
        
        def _get_orm_class(self, entity: Entity) -> Optional[Type[SQLModelType]]:
            """Get the appropriate ORM class for an entity."""
            # Try exact class match
            orm_cls = self._entity_to_orm_map.get(type(entity))
            if orm_cls:
                return orm_cls
                
            # Try parent classes
            for entity_cls, orm_cls in self._entity_to_orm_map.items():
                if isinstance(entity, entity_cls):
                    return orm_cls
                    
            return None

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
        by_id = {e.id: e for e in nodes}
        
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
            tree[entity.id] = {
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
                tree[entity.parent_id]["children"].append(entity.id)

            # Process children
            children = [e for e in nodes if e.parent_id == entity.id]
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
        nested = get_nested_entities(entity)
        for field_entities in nested.values():
            for nested_entity in field_entities:
                nested_id = id(nested_entity)
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
        cold = EntityRegistry.get_cold_snapshot(entity.id)
        
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
    if inspect.iscoroutinefunction(func):
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