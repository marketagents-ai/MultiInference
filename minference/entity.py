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
from functools import wraps

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

    def fork(self, **kwargs) -> Self:
        """Create a new version of this entity with optional data modifications."""
        # Get all fields except version-specific ones
        fields = self.model_dump(exclude={'id', 'created_at', 'lineage_id', 'parent_id'})
        
        # For lists (like history), create deep copies
        for field_name, value in fields.items():
            if isinstance(value, list):
                fields[field_name] = value.copy()
        
        # Update fields with any provided modifications
        fields.update(kwargs)
        
        # Create new version with copied fields and modifications
        new_version = self.__class__(**fields)
        new_version.parent_id = self.id
        new_version.lineage_id = self.lineage_id
        
        return new_version

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """Override model_dump to handle nested entities for comparison"""
        exclude_keys = kwargs.get('exclude', set())
        exclude_keys = exclude_keys.union({'id', 'created_at', 'lineage_id', 'parent_id'})
        kwargs['exclude'] = exclude_keys
        
        # Get base dump
        data = super().model_dump(*args, **kwargs)
        
        # Convert any nested entities to their IDs for comparison
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = [item.id if isinstance(item, Entity) else item for item in value]
            elif isinstance(value, Entity):
                data[key] = value.id
                
        return data

    @classmethod
    def compare_entities(cls, entity1: 'Entity', entity2: Dict[str, Any]) -> bool:
        """Compare an entity with a snapshot dictionary."""
        # Get current entity's data excluding version fields
        exclude_fields = {'id', 'created_at', 'lineage_id', 'parent_id', 'new_message'}
        data1 = entity1.model_dump(exclude=exclude_fields)
        
        # Compare field by field
        all_keys = set(data1.keys()) | set(entity2.keys()) - {'new_message'}
        for key in all_keys:
            val1 = data1.get(key)
            val2 = entity2.get(key)
            
            # Handle lists
            if isinstance(val1, list) and isinstance(val2, list):
                if len(val1) != len(val2):
                    return False
                for i, (item1, item2) in enumerate(zip(val1, val2)):
                    if isinstance(item1, dict) and isinstance(item2, dict):
                        if item1 != item2:
                            return False
                    elif item1 != item2:
                        return False
            # Handle regular values
            elif val1 != val2:
                return False
                
        return True

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
        """Register an entity or retrieve by UUID."""
        
        if isinstance(entity, UUID):
            return cls.get(entity)
            
        if not isinstance(entity, BaseModel):
            raise ValueError("Entity must be a Pydantic model instance")
            
        ent_id = entity.id

        # First registration
        if ent_id not in cls._snapshots:
            snapshot = entity.model_dump(exclude={'id', 'created_at', 'lineage_id', 'parent_id'})
            cls._snapshots[ent_id] = snapshot
            cls._registry[ent_id] = entity
            cls._timestamps[ent_id] = datetime.utcnow()
            
            # Track lineage for new entity
            if hasattr(entity, 'lineage_id'):
                lineage_id = entity.lineage_id
                if ent_id not in cls._lineages.get(lineage_id, []):
                    cls._lineages.setdefault(lineage_id, []).append(ent_id)
            return entity

        # Get existing entity
        existing = cls._registry[ent_id]
        snapshot = cls._snapshots[ent_id]

        # Compare with snapshot
        is_same = Entity.compare_entities(entity, snapshot)
        
        if is_same:
            return existing

        # Create new version with proper lineage
        new_version = entity.fork()
        new_snapshot = new_version.model_dump(exclude={'id', 'created_at', 'lineage_id', 'parent_id'})
        
        # Store new version
        cls._snapshots[new_version.id] = new_snapshot
        cls._registry[new_version.id] = new_version
        cls._timestamps[new_version.id] = datetime.utcnow()
        
        # Update lineage - ensure we only add once and maintain proper chain
        if hasattr(new_version, 'lineage_id'):
            lineage_id = new_version.lineage_id
            lineage = cls._lineages.setdefault(lineage_id, [])
            if new_version.id not in lineage:
                # Find the latest version in this lineage
                latest_version_id = lineage[-1] if lineage else None
                if latest_version_id:
                    # Set parent to latest version
                    new_version.parent_id = latest_version_id
                lineage.append(new_version.id)
                
        return new_version

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
        
        # Sort items by creation time to show progression
        sorted_items = sorted(tree.items(), key=lambda kv: kv[1]["created_at"])
        
        mermaid = ["graph TD"]
        
        # First add all nodes with their diffs
        for i, (vid, node) in enumerate(sorted_items):
            entity = node["entity"]
            if i > 0:  # Skip first node as it has no parent to compare with
                parent_id = node["parent_id"]
                if parent_id:
                    parent = cls.get(parent_id)
                    if parent:
                        # Compare with parent to find differences
                        parent_snapshot = parent.model_dump(exclude={'id', 'created_at', 'lineage_id', 'parent_id'})
                        current_snapshot = entity.model_dump(exclude={'id', 'created_at', 'lineage_id', 'parent_id'})
                        
                        # Find what changed
                        changes = []
                        for key in current_snapshot:
                            if key in parent_snapshot:
                                if isinstance(current_snapshot[key], list):
                                    if len(current_snapshot[key]) != len(parent_snapshot[key]):
                                        changes.append(f"{key}: {len(parent_snapshot[key])}→{len(current_snapshot[key])}")
                                elif current_snapshot[key] != parent_snapshot[key]:
                                    changes.append(f"{key}: {parent_snapshot[key]}→{current_snapshot[key]}")
                        
                        change_str = "<br/>" + "<br/>".join(changes) if changes else ""
                        mermaid.append(f"    {str(vid)}[{type(entity).__name__}{change_str}]")
                    else:
                        mermaid.append(f"    {str(vid)}[{type(entity).__name__}]")
            else:
                # First node
                mermaid.append(f"    {str(vid)}[{type(entity).__name__}]")
        
        # Then add edges based on parent relationships
        for vid, node in sorted_items:
            if node["parent_id"]:
                mermaid.append(f"    {str(node['parent_id'])} --> {str(vid)}")
        
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

def entity_uuid_expander(param_name: str):
    """Decorator to handle entity UUID expansion and versioning."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get the entity from args or kwargs
            entity = kwargs.get(param_name)
            if entity is None and len(args) > 0:
                entity = args[0]
            
            if not isinstance(entity, Entity):
                raise ValueError(f"Parameter {param_name} must be an Entity instance")

            # Get latest version from registry
            latest = EntityRegistry.get(entity.id)
            if latest and latest.id != entity.id:
                # Update args or kwargs with latest version
                if param_name in kwargs:
                    kwargs[param_name] = latest
                else:
                    args = list(args)
                    args[0] = latest
                    args = tuple(args)

            # Call the original function
            result = await func(*args, **kwargs)

            # After function call, check if entity was modified
            current = EntityRegistry.get(latest.id if latest else entity.id)
            if current:
                # Get the snapshot of the original entity for comparison
                original_snapshot = entity.model_dump(exclude={'id', 'created_at', 'lineage_id', 'parent_id'})
                
                # Compare current entity with original snapshot
                if not Entity.compare_entities(current, original_snapshot):
                    # Create new version with current as parent
                    new_version = current.fork()
                    new_version.parent_id = current.id
                    
                    # Register new version
                    EntityRegistry.register(new_version)
                    
                    # Update result if it's the modified entity
                    if result == current:
                        result = new_version

            return result
        return wrapper
    return decorator

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
