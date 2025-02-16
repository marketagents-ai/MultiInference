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

# T_Entity will represent an Entity or its subclass.
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
        """Automatically register this entity after creation/update."""
        from __main__ import EntityRegistry
        EntityRegistry._logger.info(f"{self.__class__.__name__}({self.id}): Registering entity")
        try:
            # Call our register function – note that it updates the working object if needed.
            EntityRegistry.register(self)
            EntityRegistry._logger.info(f"{self.__class__.__name__}({self.id}): Successfully registered")
        except Exception as exc:
            EntityRegistry._logger.error(f"{self.__class__.__name__}({self.id}): Registration failed - {exc}")
            raise ValueError(f"Entity registration failed: {exc}") from exc
        return self

    def _custom_serialize(self) -> Dict[str, Any]:
        """Hook for subclasses to store extra data during save()."""
        return {}

    @classmethod
    def _custom_deserialize(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for subclasses to rehydrate extra data during load()."""
        return {}

    def save(self, path: Path) -> None:
        """Save the entity to a file as JSON."""
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
                raise ValueError(f"Entity type mismatch. File has {metadata['entity_type']}, expected {cls.__name__}")
            instance = cls(**{**data_base, **data_custom})
            EntityRegistry._logger.info(f"{cls.__name__}({instance.id}): Successfully loaded")
            return instance
        except Exception as exc:
            EntityRegistry._logger.error(f"{cls.__name__}: Load failed - {exc}")
            raise IOError(f"Failed to load entity: {exc}") from exc

    @classmethod
    def get(cls: Type[T_Self], entity_id: UUID) -> Optional[T_Self]:
        """Retrieve an entity by its ID."""
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
        """Retrieve multiple entities by their IDs."""
        from __main__ import EntityRegistry
        return EntityRegistry.get_many(entity_ids, expected_type=cls)

    def fork(self: T_Self, **kwargs: Any) -> T_Self:
        """
        Create a new copy of this entity with updated fields.
        A new id is generated and the parent's id is recorded.
        """
        version_fields = {
            "parent_id": self.id,
            "id": uuid4(),
            "created_at": datetime.utcnow()
        }
        version_fields.update(kwargs)
        temp_copy = self.model_copy(update=version_fields)
        data = temp_copy.model_dump()
        new_entity = self.__class__.model_validate(data)
        # The new entity will auto-register via its validator.
        return new_entity

########################################
# 3) Snapshot-based EntityRegistry
########################################

EType = TypeVar('EType', bound=Entity)

class EntityRegistry(BaseRegistry[EType]):
    """
    Registry for managing immutable Pydantic model instances with lineage-based versioning.

    When an entity is re-registered and its non-unique fields have changed,
    a new version is forked from the working object (not from the original snapshot).
    The working object's __dict__ is then updated to match the new version,
    so subsequent calls build on that new snapshot.
    
    This method accepts both Entity instances and UUIDs.
    """
    _registry: Dict[UUID, EType] = {}
    _timestamps: Dict[UUID, datetime] = {}
    _inference_orchestrator: Optional[object] = None

    # lineage_id -> [all version UUIDs]
    _lineages: Dict[UUID, List[UUID]] = {}
    # entity_id -> snapshot dict (excluding unique fields)
    _snapshots: Dict[UUID, Dict[str, Any]] = {}

    @classmethod
    def register(cls, entity: Union[EType, UUID]) -> Optional[EType]:
        """
        Register an entity or, if a UUID is passed in, return the associated entity.
        
        If the entity's non-unique fields have changed since the last registration,
        a new version is forked from the working object. The working object's __dict__
        is updated to reflect the new version. This ensures a linear chain of versions.
        """
        # If a UUID is passed in, simply retrieve the entity.
        if isinstance(entity, UUID):
            return cls.get(entity)
        if not isinstance(entity, BaseModel):
            cls._logger.error(f"Invalid entity type: {type(entity)}")
            raise ValueError("Entity must be a Pydantic model instance")
        if not hasattr(entity, 'id'):
            cls._logger.error(f"Entity missing ID field: {type(entity)}")
            raise ValueError("Entity must have an 'id' field")

        ent_id = entity.id
        cls._logger.debug(f"Registering {entity.__class__.__name__}({ent_id})")
        exclude_keys = {'id', 'created_at', 'lineage_id', 'parent_id'}
        new_data = entity.model_dump(exclude=exclude_keys)

        # First registration.
        if ent_id not in cls._snapshots:
            cls._snapshots[ent_id] = new_data
            cls._registry[ent_id] = entity
            cls._timestamps[ent_id] = datetime.utcnow()
            cls._logger.info(f"Registered new {entity.__class__.__name__}({ent_id})")
            if hasattr(entity, 'lineage_id'):
                lineage_val = getattr(entity, 'lineage_id')
                if isinstance(lineage_val, UUID):
                    cls._lineages.setdefault(lineage_val, []).append(ent_id)
            return entity

        # Compare snapshot.
        old_data = cls._snapshots[ent_id]
        if old_data == new_data:
            cls._logger.debug(f"{entity.__class__.__name__}({ent_id}) unchanged.")
            return entity

        # Detect and log specific changes
        changes = []
        for key in set(old_data.keys()) | set(new_data.keys()):
            old_val = old_data.get(key)
            new_val = new_data.get(key)
            if old_val != new_val:
                changes.append(f"{key}: {old_val!r} -> {new_val!r}")

        cls._logger.info(
            f"Detected changes in {entity.__class__.__name__}({ent_id}); forking new version.\n"
            f"Changes detected:\n" + "\n".join(f"  - {change}" for change in changes)
        )
        changed_kwargs = {k: newval for k, newval in new_data.items() if newval != old_data.get(k)}
        new_entity = entity.fork(**changed_kwargs)
        new_id = new_entity.id
        cls._registry[new_id] = new_entity
        cls._timestamps[new_id] = datetime.utcnow()
        cls._snapshots[new_id] = new_entity.model_dump(exclude=exclude_keys)
        if hasattr(new_entity, 'lineage_id'):
            lineage_val = getattr(new_entity, 'lineage_id')
            cls._lineages.setdefault(lineage_val, []).append(new_id)
        cls._logger.info(f"Created new version: {new_entity.__class__.__name__}({new_id})")
        # Update the working object's state (in-place) to match the new version.
        entity.__dict__.update(new_entity.__dict__)
        return entity

    @classmethod
    def get(cls, entity_id: UUID, expected_type: Optional[Type[EType]] = None) -> Optional[EType]:
        cls._logger.debug(f"Retrieving entity {entity_id}")
        entity = cls._registry.get(entity_id)
        if entity is None:
            cls._logger.debug(f"Entity {entity_id} not found")
            return None
        if expected_type and not isinstance(entity, expected_type):
            cls._logger.error(f"Type mismatch for {entity_id}. Expected {expected_type.__name__}, got {type(entity).__name__}")
            return None
        return entity

    @classmethod
    def list_by_type(cls, entity_type: Type[EType]) -> List[EType]:
        cls._logger.debug(f"Listing entities of type {entity_type.__name__}")
        return [e for e in cls._registry.values() if isinstance(e, entity_type)]

    @classmethod
    def get_many(cls, entity_ids: List[UUID], expected_type: Optional[Type[EType]] = None) -> List[EType]:
        cls._logger.debug(f"Retrieving {len(entity_ids)} entities")
        results = []
        for uid in entity_ids:
            ent = cls.get(uid, expected_type=expected_type)
            if ent is not None:
                results.append(ent)
        return results

    @classmethod
    def get_registry_status(cls) -> Dict[str, Any]:
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
        cls._inference_orchestrator = inference_orchestrator

    @classmethod
    def get_inference_orchestrator(cls) -> Optional[object]:
        return cls._inference_orchestrator

    @classmethod
    def get_lineage_ids(cls, lineage_id: UUID) -> List[UUID]:
        return cls._lineages.get(lineage_id, [])

    @classmethod
    def get_lineage_tree_sorted(cls, lineage_id: UUID) -> str:
        tree = cls.build_lineage_tree(lineage_id)
        if not tree:
            return "No lineage found"
        sorted_items = sorted(tree.items(), key=lambda kv: kv[1]["created_at"])
        mermaid = ["graph TD"]
        for vid, node in sorted_items:
            entity = node["entity"]
            if hasattr(entity, 'some_data'):
                label = f"{str(vid)[:8]}[{entity.some_data}]"
            else:
                label = f"{str(vid)[:8]}[{type(entity).__name__}]"
            mermaid.append(f"    {str(vid)[:8]}{label}")
        for vid, node in tree.items():
            if node["parent_id"]:
                mermaid.append(f"    {str(node['parent_id'])[:8]} --> {str(vid)[:8]}")
        return "\n".join(mermaid)

    @classmethod
    def build_lineage_tree(cls, lineage_id: UUID) -> Dict[UUID, Dict[str, Any]]:
        version_ids = cls._lineages.get(lineage_id, [])
        if not version_ids:
            cls._logger.info(f"No versions found for lineage {lineage_id}")
            return {}
        tree: Dict[UUID, Dict[str, Any]] = {}
        for vid in version_ids:
            entity = cls._registry.get(vid)
            if not isinstance(entity, Entity):
                continue
            parent_id = getattr(entity, 'parent_id', None)
            if parent_id and parent_id not in version_ids:
                parent_id = None
            tree[vid] = {
                "entity": entity,
                "children": [],
                "depth": 1,
                "parent_id": parent_id,
                "created_at": entity.created_at
            }
        for vid, node in tree.items():
            pid = node["parent_id"]
            if pid and pid in tree:
                tree[pid]["children"].append(vid)
        for vid, node in tree.items():
            if node["parent_id"] is None:
                node["depth"] = 0
        return tree

########################################
# 4) Decorators
########################################

# For the decorators we now simply call register.
# (Since register handles UUID inputs and updates the working object.)
T_dec = TypeVar("T_dec")
PS = ParamSpec("PS")
RT = TypeVar("RT")

def entity_uuid_expander(func: Callable[[T_Entity], RT]) -> Callable[[Union[UUID, T_Entity]], RT]:
    """
    Decorator for functions that accept a single entity (or UUID).
    
    It simply calls EntityRegistry.register() on the input before and after
    executing the function, so that the working object's state is updated.
    """
    @functools.wraps(func)
    def wrapper(x: Union[UUID, T_Entity]) -> RT:
        from __main__ import EntityRegistry
        # If x is a UUID, convert it.
        if isinstance(x, UUID):
            entity : Union[T_Entity, None] = EntityRegistry.get(x)
            if entity is None:
                raise ValueError(f"No entity found for UUID: {x}")
        else:
            entity = x
        # Call register to update the working object.
        EntityRegistry.register(entity)
        result = func(entity)
        EntityRegistry.register(entity)
        return result
    return wrapper

def entity_uuid_expander_list_sync(param_name: str) -> Callable[[Callable[PS, RT]], Callable[PS, RT]]:
    """
    Synchronous decorator factory for functions accepting a list (named `param_name`)
    of entities (or UUIDs). Each element is processed via register() to update its state.
    """
    def decorator(func: Callable[PS, RT]) -> Callable[PS, RT]:
        @functools.wraps(func)
        def wrapper(*args: PS.args, **kwargs: PS.kwargs) -> RT:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            if param_name in bound.arguments:
                items = bound.arguments[param_name]
                if isinstance(items, list):
                    # Convert UUIDs to entities where needed.
                    from __main__ import EntityRegistry
                    processed = []
                    for item in items:
                        if isinstance(item, UUID):
                            ent = EntityRegistry.get(item)
                            if ent is None:
                                raise ValueError(f"No entity found for UUID: {item}")
                            processed.append(ent)
                        else:
                            processed.append(item)
                    bound.arguments[param_name] = processed
                    for item in processed:
                        EntityRegistry.register(item)
            result = func(*bound.args, **bound.kwargs)
            # After call, update each item.
            if param_name in bound.arguments:
                for item in bound.arguments[param_name]:
                    EntityRegistry.register(item)
            return result
        return wrapper
    return decorator

def entity_uuid_expander_list_async(param_name: str) -> Callable[[Callable[PS, Awaitable[RT]]], Callable[PS, Awaitable[RT]]]:
    """
    Asynchronous decorator factory for functions accepting a list (named `param_name`)
    of entities (or UUIDs). It converts UUIDs to entities and calls register() on each.
    """
    def decorator(func: Callable[PS, Awaitable[RT]]) -> Callable[PS, Awaitable[RT]]:
        @functools.wraps(func)
        async def wrapper(*args: PS.args, **kwargs: PS.kwargs) -> RT:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            if param_name in bound.arguments:
                items = bound.arguments[param_name]
                if isinstance(items, list):
                    from __main__ import EntityRegistry
                    processed = []
                    for item in items:
                        if isinstance(item, UUID):
                            ent = EntityRegistry.get(item)
                            if ent is None:
                                raise ValueError(f"No entity found for UUID: {item}")
                            processed.append(ent)
                        else:
                            processed.append(item)
                    bound.arguments[param_name] = processed
                    for item in processed:
                        EntityRegistry.register(item)
            result = await func(*bound.args, **bound.kwargs)
            if param_name in bound.arguments:
                for item in bound.arguments[param_name]:
                    EntityRegistry.register(item)
            return result
        return wrapper
    return decorator

def entity_uuid_expander_list(param_name: str) -> Any:
    """
    Decorator factory that dispatches between synchronous and asynchronous
    list decorators based on the wrapped function.
    """
    def decorator(func: Any) -> Any:
        if inspect.iscoroutinefunction(func):
            return entity_uuid_expander_list_async(param_name)(func)
        return entity_uuid_expander_list_sync(param_name)(func)
    return decorator
