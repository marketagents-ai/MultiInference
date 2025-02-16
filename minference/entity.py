"""
Registry implementation for managing immutable Pydantic model instances with lineage-based auto-versioning.

Backwards Compatibility:
- The 'Entity' class still has the same register/save/load methods, plus optional
  'lineage_id' and 'parent_id' fields for version tracking.
- The 'EntityRegistry' class still has the same methods (register, get, list_by_type, get_many, etc.).
  If the same id is re-registered with changed fields, it spawns a new child version in place.
- A snapshot mechanism ensures in-place changes are detected even if the same Python object is used.

Additionally:
- Decorators:
  - `@entity_uuid_expander` handles a single parameter typed T but allows passing either T or UUID.
  - `@entity_uuid_expander_list(param_name)` handles a parameter typed List[T] but allows passing
    either List[T] or List[UUID].
"""

import json
import inspect
import functools
import logging
from typing import (
    Dict, Any, Optional, Type, TypeVar, List, Protocol, runtime_checkable,
    Union, Callable, overload, get_args, ParamSpec, TypeAlias, Awaitable, cast, Self
)
from uuid import UUID, uuid4
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

# This is your original base registry import
from minference.base_registry import BaseRegistry


########################################
# 1) Protocol + Generics
########################################

@runtime_checkable
class HasID(Protocol):
    """Protocol requiring an `id: UUID` field."""
    id: UUID

# We'll define T=Entity below
T_Entity = TypeVar('T_Entity', bound='Entity')


########################################
# 2) The Entity class
########################################

T_Self = TypeVar('T_Self', bound='Entity')

class Entity(BaseModel):
    """
    Base class for registry-integrated, serializable entities.

    Subclasses are responsible for custom serialization logic, 
    possibly nested relationships, etc.

    Snapshots + re-registration => auto-versioning if fields change in place.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this entity instance (version).")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when entity was created.")

    lineage_id: UUID = Field(default_factory=uuid4, description="Stable ID for entire lineage of versions.")
    parent_id: Optional[UUID] = Field(default=None, description="If set, points to parent's version ID.")

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda dt: dt.isoformat()
        }

    @model_validator(mode='after')
    def register_entity(self) -> Self:
        """Register in the global registry after model init or update."""
        from __main__ import EntityRegistry
        EntityRegistry._logger.info(f"{self.__class__.__name__}({self.id}): Registering entity")
        try:
            EntityRegistry.register(self)
            EntityRegistry._logger.info(f"{self.__class__.__name__}({self.id}): Successfully registered")
        except Exception as exc:
            EntityRegistry._logger.error(f"{self.__class__.__name__}({self.id}): Registration failed - {exc}")
            raise ValueError(f"Entity registration failed: {exc}") from exc
        return self

    def _custom_serialize(self) -> Dict[str, Any]:
        """
        Hook for subclasses to store custom data during save().
        Return a dictionary that merges into the final JSON under 'custom_data'.
        """
        return {}

    @classmethod
    def _custom_deserialize(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook for subclasses to handle custom data during load().
        'data' is what was stored in 'custom_data' field.
        Return a dict of fields to pass into the constructor.
        """
        return {}

    def save(self, path: Path) -> None:
        """Save this entity instance to a file as JSON."""
        from __main__ import EntityRegistry
        EntityRegistry._logger.info(f"{self.__class__.__name__}({self.id}): Saving to {path}")

        try:
            base_data = self.model_dump()
            metadata = {
                "entity_type": self.__class__.__name__,
                "schema_version": "1.0",
                "saved_at": datetime.utcnow().isoformat()
            }
            custom_data = self._custom_serialize()

            full_json = {
                "metadata": metadata,
                "data": base_data,
                "custom_data": custom_data
            }
            with open(path, 'w') as f:
                json.dump(full_json, f, indent=2)
            EntityRegistry._logger.info(f"{self.__class__.__name__}({self.id}): Successfully saved")
        except Exception as exc:
            EntityRegistry._logger.error(f"{self.__class__.__name__}({self.id}): Save failed - {exc}")
            raise IOError(f"Failed to save entity: {exc}") from exc

    @classmethod
    def load(cls: Type['Entity'], path: Path) -> 'Entity':
        """Load an entity instance from a JSON file."""
        from __main__ import EntityRegistry
        EntityRegistry._logger.info(f"{cls.__name__}: Loading from {path}")

        try:
            with open(path, 'r') as f:
                all_json = json.load(f)
            metadata = all_json["metadata"]
            data_base = all_json["data"]
            data_custom = cls._custom_deserialize(all_json.get("custom_data", {}))

            if metadata["entity_type"] != cls.__name__:
                raise ValueError(
                    f"Entity type mismatch. File has {metadata['entity_type']}, expected {cls.__name__}"
                )

            instance = cls(**{**data_base, **data_custom})
            EntityRegistry._logger.info(f"{cls.__name__}({instance.id}): Successfully loaded")
            return instance
        except Exception as exc:
            EntityRegistry._logger.error(f"{cls.__name__}: Load failed - {exc}")
            raise IOError(f"Failed to load entity: {exc}") from exc

    @classmethod
    def get(cls: Type[T_Self], entity_id: UUID) -> Optional[T_Self]:
        """Get an entity from the registry by its ID."""
        from __main__ import EntityRegistry
        entity = EntityRegistry.get(entity_id, expected_type=cls)
        return cast(Optional[T_Self], entity)

    @classmethod
    def list_all(cls: Type['Entity']) -> List['Entity']:
        """List all entities of this type."""
        from __main__ import EntityRegistry
        return EntityRegistry.list_by_type(cls)

    @classmethod
    def get_many(cls: Type['Entity'], entity_ids: List[UUID]) -> List['Entity']:
        """Get multiple entities by their IDs."""
        from __main__ import EntityRegistry
        return EntityRegistry.get_many(entity_ids, expected_type=cls)

    def fork(self: T_Self, **kwargs: Any) -> T_Self:
        """
        Create a new copy of this entity with optional field updates.
        The new copy is automatically registered with a new ID and lineage tracking.
        """
        # Merge user kwargs with required version tracking fields
        version_fields = {
            "parent_id": self.id,
            "id": uuid4(),
            "created_at": datetime.utcnow()
        }
        # User kwargs take precedence over version fields if provided
        version_fields.update(kwargs)
        
        # Create copy with all fields at once
        temp_copy = self.model_copy(update=version_fields, deep=True)
        
        # Dump to dict and validate through a new instance
        data = temp_copy.model_dump()
        new_entity = self.__class__.model_validate(data)
        
        # Registration happens automatically via validator
        return new_entity


########################################
# 3) Snapshot-based EntityRegistry
########################################

EType = TypeVar('EType', bound=HasID)

class EntityRegistry(BaseRegistry[EType]):
    """
    Registry for managing immutable Pydantic model instances with lineage-based versioning.

    If the same ID is re-registered with changed fields, we spawn a new version by:
      - Setting parent_id = old_id
      - Generating a new id
      - Resetting created_at
    We maintain snapshots so we can detect changes, even if the user mutates
    the *same Python object* that was previously in _registry.
    """
    _registry: Dict[UUID, EType] = {}
    _timestamps: Dict[UUID, datetime] = {}
    _inference_orchestrator: Optional[object] = None

    _lineages: Dict[UUID, List[UUID]] = {}
    _snapshots: Dict[UUID, Dict[str, Any]] = {}

    @classmethod
    def register(cls, entity: EType) -> None:
        """
        Register a new entity or update if it matches an existing ID. 
        If fields differ from snapshot => spawn new version. 
        """
        if not isinstance(entity, BaseModel):
            cls._logger.error(f"Invalid entity type: {type(entity)}")
            raise ValueError("Entity must be a Pydantic model instance")
        if not hasattr(entity, 'id'):
            cls._logger.error(f"Entity missing ID field: {type(entity)}")
            raise ValueError("Entity must have an 'id' field")

        ent_id = entity.id
        cls._logger.debug(f"Attempting to register {entity.__class__.__name__}({ent_id})")

        # Build "payload" snapshot excluding ID, created_at, lineage_id, parent_id
        exclude_keys = {'id', 'created_at', 'lineage_id', 'parent_id'}
        new_data = entity.model_dump(exclude=exclude_keys)

        if ent_id not in cls._snapshots:
            # brand-new snapshot
            cls._logger.debug(f"No prior snapshot for {ent_id}; storing initial snapshot")
            cls._snapshots[ent_id] = new_data
        else:
            # compare old vs new
            old_data = cls._snapshots[ent_id]
            if old_data == new_data:
                # identical => do nothing
                cls._logger.debug(
                    f"{entity.__class__.__name__}({ent_id}) already registered and matches snapshot"
                )
                return
            else:
                # spawn a new version
                cls._logger.info(
                    f"Detected changes in re-registered {entity.__class__.__name__}({ent_id}); new version"
                )
                old_id = ent_id
                if hasattr(entity, 'parent_id'):
                    setattr(entity, 'parent_id', old_id)
                new_id = uuid4()
                setattr(entity, 'id', new_id)
                if hasattr(entity, 'created_at'):
                    setattr(entity, 'created_at', datetime.utcnow())
                ent_id = new_id

        # store in _registry, set timestamp
        cls._registry[ent_id] = entity
        cls._timestamps[ent_id] = datetime.utcnow()
        cls._logger.info(f"Successfully registered {entity.__class__.__name__}({ent_id})")

        # lineage updates
        if hasattr(entity, 'lineage_id'):
            lineage_val = getattr(entity, 'lineage_id')
            if isinstance(lineage_val, UUID):
                if lineage_val not in cls._lineages:
                    cls._lineages[lineage_val] = []
                if ent_id not in cls._lineages[lineage_val]:
                    cls._lineages[lineage_val].append(ent_id)

        # record new snapshot
        cls._snapshots[ent_id] = new_data

    @classmethod
    def get(
        cls,
        entity_id: UUID,
        expected_type: Optional[Type[EType]] = None
    ) -> Optional[EType]:
        """Retrieve an entity by ID with optional type checking."""
        cls._logger.debug(f"Retrieving entity {entity_id}")
        entity = cls._registry.get(entity_id)
        if entity is None:
            cls._logger.debug(f"Entity {entity_id} not found")
            return None
        if expected_type and not isinstance(entity, expected_type):
            cls._logger.error(
                f"Type mismatch for {entity_id}. "
                f"Expected {expected_type.__name__}, got {type(entity).__name__}"
            )
            return None
        return entity

    @classmethod
    def list_by_type(cls, entity_type: Type[EType]) -> List[EType]:
        """List all entities of a specific type."""
        cls._logger.debug(f"Listing entities of type {entity_type.__name__}")
        return [e for e in cls._registry.values() if isinstance(e, entity_type)]

    @classmethod
    def get_many(
        cls,
        entity_ids: List[UUID],
        expected_type: Optional[Type[EType]] = None
    ) -> List[EType]:
        """Get multiple entities by IDs."""
        cls._logger.debug(f"Retrieving {len(entity_ids)} entities")
        results = []
        for uid in entity_ids:
            ent = cls.get(uid, expected_type=expected_type)
            if ent is not None:
                results.append(ent)
        return results

    @classmethod
    def get_registry_status(cls) -> Dict[str, Any]:
        """Return registry status, including lineage info."""
        base_status = super().get_registry_status()
        type_counts: Dict[str, int] = {}
        for e in cls._registry.values():
            nm = e.__class__.__name__
            type_counts[nm] = type_counts.get(nm, 0) + 1

        timestamps = sorted(cls._timestamps.values())
        total_lineages = len(cls._lineages)
        total_versions = sum(len(v) for v in cls._lineages.values())

        return {
            **base_status,
            "entities_by_type": type_counts,
            "version_history": {
                "first_version": timestamps[0].isoformat() if timestamps else None,
                "latest_version": timestamps[-1].isoformat() if timestamps else None,
                "version_count": len(timestamps)
            },
            "total_lineages": total_lineages,
            "total_versions": total_versions,
        }

    @classmethod
    def set_inference_orchestrator(cls, inference_orchestrator: object) -> None:
        """Set the inference orchestrator (unchanged)."""
        cls._inference_orchestrator = inference_orchestrator

    @classmethod
    def get_inference_orchestrator(cls) -> Optional[object]:
        """Get the inference orchestrator (unchanged)."""
        return cls._inference_orchestrator

    @classmethod
    def get_lineage_ids(cls, lineage_id: UUID) -> List[UUID]:
        """Return list of version IDs in a given lineage."""
        return cls._lineages.get(lineage_id, [])

    @classmethod
    def get_lineage_tree_sorted(cls, lineage_id: UUID) -> List[EType]:
        """Return all versions in ascending created_at order."""
        version_ids = cls._lineages.get(lineage_id, [])
        results = [cls._registry[v] for v in version_ids if v in cls._registry]
        results.sort(key=lambda e: getattr(e, 'created_at', datetime.min))
        return results


########################################
# 4) Decorators
########################################

U = TypeVar("U")
Ret = TypeVar("Ret", covariant=True)

@overload
def entity_uuid_expander(func: Callable[[T_Entity], Ret]) -> Callable[[Union[UUID, T_Entity]], Ret]:
    ...

@overload
def entity_uuid_expander(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    ...

def entity_uuid_expander(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Decorator that modifies a single-arg function expecting an Entity,
    so that from the language server viewpoint, it supports (UUID or T).
    """
    @functools.wraps(func)
    def wrapper(x: Any) -> Any:
        from __main__ import EntityRegistry
        if isinstance(x, UUID):
            ent = EntityRegistry.get(x)
            if ent is None:
                raise ValueError(f"No entity found in registry for id={x}")
            EntityRegistry.register(ent)
            result = func(ent)
            EntityRegistry.register(ent)
            return result
        else:
            EntityRegistry.register(x)
            result = func(x)
            EntityRegistry.register(x)
            return result
    return wrapper


PS = ParamSpec("PS")
RT = TypeVar("RT")

AsyncCallable: TypeAlias = Callable[..., Awaitable[Any]]
SyncCallable: TypeAlias = Callable[..., Any]

def entity_uuid_expander_list_sync(param_name: str) -> Callable[[Callable[PS, RT]], Callable[PS, RT]]:
    """
    A decorator factory for synchronous functions that modifies exactly one parameter named `param_name`.
    That parameter must be typed as List[EntitySubclass].  
    """
    def decorator(func: Callable[PS, RT]) -> Callable[PS, RT]:
        @functools.wraps(func)
        def wrapper(*args: PS.args, **kwargs: PS.kwargs) -> RT:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            if param_name not in bound.arguments:
                return func(*bound.args, **bound.kwargs)

            param_value = bound.arguments[param_name]
            if not isinstance(param_value, list) or not param_value:
                return func(*bound.args, **bound.kwargs)

            from __main__ import EntityRegistry
            first_elem = param_value[0]
            resolved = []
            if isinstance(first_elem, UUID):
                for uid in param_value:
                    ent = EntityRegistry.get(uid)
                    if ent is None:
                        raise ValueError(f"No entity found for UUID={uid}")
                    EntityRegistry.register(ent)
                    resolved.append(ent)
            else:
                for e in param_value:
                    EntityRegistry.register(e)
                    resolved.append(e)

            bound.arguments[param_name] = resolved
            result = func(*bound.args, **bound.kwargs)

            for e in resolved:
                EntityRegistry.register(e)

            return result
        return wrapper
    return decorator

def entity_uuid_expander_list_async(param_name: str) -> Callable[[Callable[PS, Awaitable[RT]]], Callable[PS, Awaitable[RT]]]:
    """
    A decorator factory for async functions that modifies exactly one parameter named `param_name`.
    That parameter must be typed as List[EntitySubclass].  
    """
    def decorator(func: Callable[PS, Awaitable[RT]]) -> Callable[PS, Awaitable[RT]]:
        @functools.wraps(func)
        async def wrapper(*args: PS.args, **kwargs: PS.kwargs) -> RT:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            if param_name not in bound.arguments:
                return await func(*bound.args, **bound.kwargs)

            param_value = bound.arguments[param_name]
            if not isinstance(param_value, list) or not param_value:
                return await func(*bound.args, **bound.kwargs)

            from __main__ import EntityRegistry
            first_elem = param_value[0]
            resolved = []
            if isinstance(first_elem, UUID):
                for uid in param_value:
                    ent = EntityRegistry.get(uid)
                    if ent is None:
                        raise ValueError(f"No entity found for UUID={uid}")
                    EntityRegistry.register(ent)
                    resolved.append(ent)
            else:
                for e in param_value:
                    EntityRegistry.register(e)
                    resolved.append(e)

            bound.arguments[param_name] = resolved
            result = await func(*bound.args, **bound.kwargs)

            for e in resolved:
                EntityRegistry.register(e)

            return result
        return wrapper
    return decorator

def entity_uuid_expander_list(param_name: str) -> Any:
    """
    A decorator factory that modifies exactly one parameter named `param_name`.
    Automatically chooses between sync and async implementations.
    """
    def decorator(func: Any) -> Any:
        if inspect.iscoroutinefunction(func):
            return entity_uuid_expander_list_async(param_name)(func)
        return entity_uuid_expander_list_sync(param_name)(func)
    return decorator


