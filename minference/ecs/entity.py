############################################################
# entity.py - Refactored version with unified entity comparison
############################################################

import json
import inspect
import logging
import re
from typing import (
    Any, Dict, Optional, Type, TypeVar, List, Protocol, runtime_checkable,
    Union, Callable, get_args, cast, Self, Tuple, Set
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
# 2) Unified Comparison Function
##############################

def compare_values(v1: Any, v2: Any, path: str = "", diffs: Optional[Dict[str, Dict[str, Any]]] = None,
                  visited: Optional[Set[Tuple[int, int]]] = None) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """Enhanced comparison with better handling of SQL serialization edge cases."""
    # Initialize containers
    if diffs is None:
        diffs = {}
    if visited is None:
        visited = set()
    
    # Handle identity comparison
    if v1 is v2:
        return True, diffs
    
    # Prevent cycles 
    obj_pair = (id(v1), id(v2))
    if obj_pair in visited:
        return True, diffs
    visited.add(obj_pair)
    
    # Handle None values
    if v1 is None or v2 is None:
        if v1 is not v2:
            diffs[path] = {"type": "modified", "old": v2, "new": v1}
            return False, diffs
        return True, diffs
    
    # Handle type differences with special cases for SQL
    if type(v1) != type(v2):
        # Special case: str vs UUID (common in SQL backends)
        if (isinstance(v1, (str, UUID)) and isinstance(v2, (str, UUID))):
            try:
                if str(v1) == str(v2):
                    return True, diffs
            except:
                pass
        
        diffs[path] = {"type": "type_mismatch", "old_type": type(v2).__name__, "new_type": type(v1).__name__}
        return False, diffs
    
    # Entity comparison logic
    if isinstance(v1, Entity) and isinstance(v2, Entity):
        # Different entities
        if v1.id != v2.id:
            diffs[path] = {"type": "entity_changed", "old_id": v2.id, "new_id": v1.id}
            return False, diffs
        
        # Field comparison with exclusions
        exclude = {'id', 'created_at', 'parent_id', 'live_id', 'old_ids', 'lineage_id', 'from_storage'}
        fields1 = set(v1.model_fields.keys()) - exclude
        fields2 = set(v2.model_fields.keys()) - exclude
        
        # Check for field structure changes
        if fields1 != fields2:
            added = fields1 - fields2
            removed = fields2 - fields1
            if added:
                diffs[f"{path}.fields_added"] = {"type": "fields_added", "fields": list(added)}
            if removed:
                diffs[f"{path}.fields_removed"] = {"type": "fields_removed", "fields": list(removed)}
            return False, diffs
        
        # Compare each field
        changed = False
        for f in fields1:
            field_path = f"{path}.{f}" if path else f
            f1 = getattr(v1, f)
            f2 = getattr(v2, f)
            equal, diffs = compare_values(f1, f2, field_path, diffs, visited)
            if not equal:
                changed = True
        
        return not changed, diffs
    
    # Floating point comparison with improved tolerance for SQL
    elif isinstance(v1, (float, int)) and isinstance(v2, (float, int)):
        # Use relative tolerance for larger numbers
        if abs(float(v1)) > 1.0 or abs(float(v2)) > 1.0:
            relative_diff = abs((float(v1) - float(v2)) / max(abs(float(v1)), abs(float(v2))))
            if relative_diff <= 1e-6:  # 0.0001% difference
                return True, diffs
        # Use absolute tolerance for smaller numbers
        elif abs(float(v1) - float(v2)) <= 1e-9:
            return True, diffs
            
        diffs[path] = {"type": "modified", "old": v2, "new": v1}
        return False, diffs
    
    # Datetime comparison with SQL-friendly tolerance
    elif isinstance(v1, datetime) and isinstance(v2, datetime):
        # Use 1 second tolerance for SQL serialization differences
        if abs((v1 - v2).total_seconds()) <= 1.0:
            return True, diffs
        diffs[path] = {"type": "modified", "old": v2, "new": v1}
        return False, diffs
    
    # String comparison with normalization for SQL
    elif isinstance(v1, str) and isinstance(v2, str):
        # Normalize whitespace and line endings
        norm_v1 = v1.strip().replace('\r\n', '\n')
        norm_v2 = v2.strip().replace('\r\n', '\n')
        if norm_v1 == norm_v2:
            return True, diffs
            
        diffs[path] = {"type": "modified", "old": v2, "new": v1}
        return False, diffs
    
    # Dictionary comparison
    elif isinstance(v1, dict) and isinstance(v2, dict):
        keys1 = set(v1.keys())
        keys2 = set(v2.keys())
        
        changed = False
        
        # Check for added/removed keys
        if keys1 != keys2:
            added = keys1 - keys2
            removed = keys2 - keys1
            
            if added:
                diffs[f"{path}.keys_added"] = {"type": "keys_added", "keys": list(added)}
                changed = True
            if removed:
                diffs[f"{path}.keys_removed"] = {"type": "keys_removed", "keys": list(removed)}
                changed = True
                
        # Check each key present in both
        for k in keys1 & keys2:
            key_path = f"{path}.{k}" if path else k
            equal, diffs = compare_values(v1[k], v2[k], key_path, diffs, visited)
            if not equal:
                changed = True
                
        return not changed, diffs
    
    # List/tuple comparison
    elif isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
        if len(v1) != len(v2):
            diffs[path] = {"type": "length_mismatch", "old_len": len(v2), "new_len": len(v1)}
            return False, diffs
            
        # For lists of entities, try to match by ID
        if v1 and v2 and all(isinstance(x, Entity) for x in v1) and all(isinstance(x, Entity) for x in v2):
            entities1 = {x.id: x for x in v1}
            entities2 = {x.id: x for x in v2}
            
            if set(entities1.keys()) != set(entities2.keys()):
                diffs[path] = {"type": "entity_set_changed", "old_ids": list(entities2.keys()), "new_ids": list(entities1.keys())}
                return False, diffs
                
            changed = False
            for eid, entity1 in entities1.items():
                entity_path = f"{path}[{eid}]"
                equal, diffs = compare_values(entity1, entities2[eid], entity_path, diffs, visited)
                if not equal:
                    changed = True
                    
            return not changed, diffs
            
        # For normal lists, compare items in order
        changed = False
        for i, (item1, item2) in enumerate(zip(v1, v2)):
            item_path = f"{path}[{i}]"
            equal, diffs = compare_values(item1, item2, item_path, diffs, visited)
            if not equal:
                changed = True
                
        return not changed, diffs
    
    # Default equality check for other types
    elif v1 == v2:
        return True, diffs
        
    # Values are different
    diffs[path] = {"type": "modified", "old": v2, "new": v1}
    return False, diffs

##############################
# 3) The Entity + Diff
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

class Entity(BaseModel):
    """
    Base class for registry-integrated, serializable entities.
    Subclasses are responsible for custom serialization logic,
    possibly nested relationships, etc.
    Snapshots + re-registration => auto-versioning if fields change in place.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    live_id: UUID = Field(default_factory=uuid4, description="Live/warm identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    parent_id: Optional[UUID] = None
    lineage_id: UUID = Field(default_factory=uuid4)
    old_ids: List[UUID] = Field(default_factory=list)
    from_storage: bool = Field(default=False, description="Whether the entity was loaded from storage")
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda dt: dt.isoformat()
        }

    def register_entity(self: "Entity") -> "Entity":
        """Register this entity with the EntityRegistry."""
        from __main__ import EntityRegistry
        if not EntityRegistry.has_entity(self.id):
            EntityRegistry.register(self)
        elif not self.from_storage:
            cold = EntityRegistry.get_cold_snapshot(self.id)
            if cold:
                is_equal, _ = compare_values(self, cold)
                if not is_equal:
                    self.fork()
        return self

    @model_validator(mode='after')
    def _auto_register(self) -> Self:
        """Auto-register entity during initialization."""
        entity = self.register_entity()
        return self

    def has_modifications(self, other: "Entity") -> bool:
        """Check if entity has modifications compared to another entity."""
        from __main__ import EntityRegistry
        EntityRegistry._logger.debug(f"Checking modifications between {self.id} and {other.id}")
        is_equal, _ = compare_values(self, other)
        return not is_equal

    def compute_diff(self, other: "Entity") -> EntityDiff:
        """Compute a detailed difference structure between entities."""
        from __main__ import EntityRegistry
        EntityRegistry._logger.debug(f"Computing diff: {self.id} vs {other.id}")
        
        is_equal, diffs = compare_values(self, other)
        
        # Convert to EntityDiff format
        result = EntityDiff()
        for path, diff_info in diffs.items():
            # Parse the path into field name
            field = path.split('.')[0] if '.' in path else path
            # Remove brackets if present (for list indices)
            field = field.split('[')[0] if '[' in field else field
            
            result.add_diff(field, diff_info["type"], 
                          old_value=diff_info.get("old"), 
                          new_value=diff_info.get("new"))
        return result

    def fork(self, force: bool = False, diff_info: Optional[Dict[str, Dict[str, Any]]] = None, **kwargs: Any) -> "Entity":
        """
        Fork an entity with proper handling of nested entities.
        
        Args:
            force: Whether to force forking even without changes
            diff_info: Optional diff information from previous comparison
            **kwargs: Additional attributes to set on the new fork
        """
        from __main__ import EntityRegistry
        cold = EntityRegistry.get_cold_snapshot(self.id)
        if cold is None:
            EntityRegistry.register(self)
            return self
            
        # Check modifications if we don't have diff_info yet
        if diff_info is None:
            is_equal, diff_info = compare_values(self, cold)
            changed = not is_equal
        else:
            changed = bool(diff_info)
            
        # Apply any explicit changes
        if kwargs:
            changed = True
            for k, v in kwargs.items():
                setattr(self, k, v)
                
        # No changes detected
        if not force and not changed:
            return self
            
        # Create the new fork with a new ID
        old_id = self.id
        self.id = uuid4()
        self.parent_id = old_id
        self.old_ids.append(old_id)
        
        # Handle nested entities that need forking
        if diff_info:
            self._fork_nested_entities(diff_info)
            
        # Register the new version
        EntityRegistry.register(self)
        return self
    
    def _fork_nested_entities(self, diff_info: Dict[str, Dict[str, Any]]) -> None:
        """
        Fork nested entities and update references to them.
        Uses better error handling and detailed logging.
        """
        from __main__ import EntityRegistry
        # Group diffs by top-level field
        field_diffs = {}
        for path, info in diff_info.items():
            field = path.split('.')[0] if '.' in path else path
            # Remove brackets if present (for list indices)
            field = field.split('[')[0] if '[' in field else field
            
            if field not in field_diffs:
                field_diffs[field] = {}
            field_diffs[field][path] = info
        
        # Process each changed field with error handling
        failures = {}
        
        # Process each changed field
        for field, field_diff in field_diffs.items():
            try:
                value = getattr(self, field, None)
                
                # Fork nested entity
                if isinstance(value, Entity):
                    EntityRegistry._logger.debug(f"Processing nested entity field {field} of {type(self).__name__}({self.id})")
                    # Only fork if there are nested changes beyond just the top path
                    nested_changes = {p: d for p, d in field_diff.items() if p != field}
                    if nested_changes:
                        EntityRegistry._logger.info(f"Forking nested entity {type(value).__name__}({value.id}) in {field}")
                        forked_entity = value.fork(diff_info=nested_changes)
                        # Update the reference in parent to the new entity version
                        setattr(self, field, forked_entity)
                        EntityRegistry._logger.debug(f"Updated reference in {type(self).__name__}.{field}: {value.id} → {forked_entity.id}")
                        
                # Handle list of entities
                elif isinstance(value, list) and value and all(isinstance(x, Entity) for x in value):
                    EntityRegistry._logger.debug(f"Processing entity list field {field} with {len(value)} items")
                    # Find which entities in the list have changes
                    entity_changes = {}
                    for path, info in field_diff.items():
                        if '[' in path:  # List index path
                            match = re.search(r'\[(.*?)\]', path)
                            if match:
                                entity_id = match.group(1)
                                try:
                                    # Try UUID parsing
                                    if len(entity_id) > 8:  # Likely a UUID
                                        # Find entity by ID
                                        for i, entity in enumerate(value):
                                            if str(entity.id) == entity_id:
                                                if i not in entity_changes:
                                                    entity_changes[i] = {}
                                                entity_changes[i][path.replace(f"[{entity_id}]", "")] = info
                                    # Try index parsing
                                    else:
                                        idx = int(entity_id)
                                        if idx not in entity_changes:
                                            entity_changes[idx] = {}
                                        entity_changes[idx][path.replace(f"[{idx}]", "")] = info
                                except ValueError:
                                    EntityRegistry._logger.warning(f"Could not parse index from path: {path}")
                    
                    # Fork each changed entity
                    for idx, changes in entity_changes.items():
                        if idx < len(value):
                            entity = value[idx]
                            EntityRegistry._logger.info(f"Forking list item {idx} ({type(entity).__name__}({entity.id}))")
                            forked_entity = entity.fork(diff_info=changes)
                            # Update reference in the list
                            value[idx] = forked_entity
                            EntityRegistry._logger.debug(f"Updated reference in {type(self).__name__}.{field}[{idx}]: {entity.id} → {forked_entity.id}")
            except Exception as e:
                EntityRegistry._logger.error(f"Failed to fork nested entity {field}: {e}")
                failures[field] = str(e)
                # Continue with other fields
                    
        if failures:
            EntityRegistry._logger.warning(f"Partial failure during nested entity forking: {failures}")

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
    def compare_entities(cls, e1: "Entity", snapshot: Dict[str, Any]) -> bool:
        """Compare entity with a dictionary snapshot."""
        # Convert entity to comparable structure
        data1 = e1.entity_dump()
        # Use our unified comparison
        is_equal, _ = compare_values(data1, snapshot)
        return is_equal

    @classmethod
    def get(cls: Type["Entity"], entity_id: UUID) -> Optional["Entity"]:
        from __main__ import EntityRegistry
        ent = EntityRegistry.get(entity_id, expected_type=cls)
        return cast(Optional["Entity"], ent)

    @classmethod
    def list_all(cls: Type["Entity"]) -> List["Entity"]:
        from __main__ import EntityRegistry
        return EntityRegistry.list_by_type(cls)

    @classmethod
    def get_many(cls: Type["Entity"], ids: List[UUID]) -> List["Entity"]:
        from __main__ import EntityRegistry
        return EntityRegistry.get_many(ids, expected_type=cls)

##############################
# 4) Enhanced Entity Collection and Dependency Tracking
##############################

def _collect_entities_with_dependencies(args: tuple, kwargs: dict) -> Dict[int, Tuple[Entity, List[Entity]]]:
    """
    Collect all entities with their dependencies using lists instead of sets.
    
    Returns a dictionary mapping entity memory id to a tuple of:
    (entity, list of entities it contains)
    """
    result = {}
    dep_map = {}  # {id(entity): list of entities it contains}
    visited_ids = set()  # Track visited object IDs for cycle detection
    
    def scan(obj: Any, container: Optional[Entity] = None) -> None:
        # Skip if already visited (cycle detection)
        obj_id = id(obj)
        if obj_id in visited_ids:
            return
        visited_ids.add(obj_id)
            
        if isinstance(obj, Entity):
            result[obj_id] = obj
            if container is not None:
                container_id = id(container)
                if container_id not in dep_map:
                    dep_map[container_id] = []
                # Use a list and check by ID to avoid hashability issues
                if obj not in dep_map[container_id]:
                    dep_map[container_id].append(obj)
            
            # Scan entity fields recursively
            for field_name in obj.model_fields.keys():
                try:
                    field_value = getattr(obj, field_name, None)
                    if field_value is not None:
                        scan(field_value, obj)
                except Exception as e:
                    EntityRegistry._logger.error(f"Error scanning field {field_name} of {type(obj).__name__}: {e}")
                    
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                scan(x, container)
        elif isinstance(obj, dict):
            for v in obj.values():
                scan(v, container)
    
    # Scan root arguments
    for a in args:
        scan(a)
    for v in kwargs.values():
        scan(v)
        
    # Create final result with dependencies
    return {eid: (entity, dep_map.get(eid, [])) for eid, entity in result.items()}

def _check_and_fork_with_dependencies(entities_with_deps: Dict[int, Tuple[Entity, List[Entity]]]) -> None:
    """Fork entities in dependency order (bottom-up)."""
    # Convert to adjacency list for topological sort
    graph = {id_e: {id(dep) for dep in deps} for id_e, (e, deps) in entities_with_deps.items()}
    
    # Find all leaf nodes (entities with no dependencies)
    processed = set()
    while len(processed) < len(entities_with_deps):
        # Find nodes whose dependencies are all processed (bottom-up approach)
        next_batch = set()
        for eid, deps in graph.items():
            if eid not in processed and deps.issubset(processed):
                next_batch.add(eid)
                
        # Process this batch
        for eid in next_batch:
            entity = entities_with_deps[eid][0]
            cold = EntityRegistry.get_cold_snapshot(entity.id)
            
            if cold:
                is_equal, diff_info = compare_values(entity, cold)
                if not is_equal:
                    # Fork with diff info for proper nested handling
                    forked_entity = entity.fork(diff_info=diff_info)
                    
                    # Update reference in any parent entities that contain this entity
                    for parent_id, (parent, children) in entities_with_deps.items():
                        if id(entity) in {id(child) for child in children}:
                            # Find and update the reference in the parent
                            for field_name in parent.model_fields.keys():
                                field_value = getattr(parent, field_name, None)
                                
                                # Direct reference to the entity
                                if field_value is entity:
                                    setattr(parent, field_name, forked_entity)
                                
                                # Entity in a list/collection
                                elif isinstance(field_value, list):
                                    for i, item in enumerate(field_value):
                                        if item is entity:
                                            field_value[i] = forked_entity
                    
            processed.add(eid)
            
        # Ensure we're making progress
        if not next_batch:
            # Graph has cycles, just process remaining entities
            for eid, (entity, _) in entities_with_deps.items():
                if eid not in processed:
                    cold = EntityRegistry.get_cold_snapshot(entity.id)
                    if cold:
                        is_equal, diff_info = compare_values(entity, cold)
                        if not is_equal:
                            entity.fork(diff_info=diff_info)
                    processed.add(eid)

##############################
# 5) Entity Collection helpers
##############################

def _collect_entities(args: tuple, kwargs: dict) -> Dict[int, Entity]:
    """
    Simpler entity collection function that doesn't track dependencies.
    Used as a fallback.
    """
    found: Dict[int, Entity] = {}
    def scan(obj: Any) -> None:
        if isinstance(obj, Entity):
            found[id(obj)] = obj
        elif isinstance(obj, (list, tuple, set)):
            for x in obj:
                scan(x)
        elif isinstance(obj, dict):
            for v in obj.values():
                scan(v)
    for a in args:
        scan(a)
    for v in kwargs.values():
        scan(v)
    return found

def _check_and_fork_modified(entity: Entity) -> None:
    """
    Simple check and fork for a single entity.
    This is a fallback for backward compatibility.
    """
    from __main__ import EntityRegistry
    cold = EntityRegistry.get_cold_snapshot(entity.id)
    if cold:
        is_equal, diff_info = compare_values(entity, cold)
        if not is_equal:
            entity.fork(diff_info=diff_info)


##############################
# 6) EntityStorage Protocol
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
    # Methods to unify lineage logic externally
    def get_lineage_entities(self, lineage_id: UUID) -> List[Entity]: ...
    def has_lineage_id(self, lineage_id: UUID) -> bool: ...
    def get_lineage_ids(self, lineage_id: UUID) -> List[UUID]: ...


##############################
# 7) InMemoryEntityStorage
##############################
class InMemoryEntityStorage(EntityStorage):
    """
    Refined in-memory storage with a _entity_class_map if desired. 
    (It's somewhat redundant in memory, because we already store the entire object.)
    """
    def __init__(self) -> None:
        self._logger = logging.getLogger("InMemoryEntityStorage")
        self._registry: Dict[UUID, Entity] = {}
        self._entity_class_map: Dict[UUID, Type[Entity]] = {}
        self._lineages: Dict[UUID, List[UUID]] = {}
        self._inference_orchestrator: Optional[object] = None

    def has_entity(self, entity_id: UUID) -> bool:
        return entity_id in self._registry

    def get_cold_snapshot(self, entity_id: UUID) -> Optional[Entity]:
        return self._registry.get(entity_id)

    def register(self, entity_or_id: Union[Entity, UUID]) -> Optional[Entity]:
        """Register entity with improved diffing and forking."""
        if isinstance(entity_or_id, UUID):
            return self.get(entity_or_id, None)

        e = entity_or_id
        
        # If from_storage, no need to check modifications (already registered)
        if getattr(e, 'from_storage', False):
            return e
            
        old = self._registry.get(e.id)
        if not old:
            self._store_cold_snapshot(e)
            return e

        # Check for modifications using unified comparison function
        is_equal, diff_info = compare_values(e, old)
        if not is_equal:
            e = e.fork(diff_info=diff_info)
            
        return e

    def get(self, entity_id: UUID, expected_type: Optional[Type[Entity]]) -> Optional[Entity]:
        ent = self._registry.get(entity_id)
        if not ent:
            return None
        if expected_type and not isinstance(ent, expected_type):
            self._logger.error(f"Type mismatch: got {type(ent).__name__}, expected {expected_type.__name__}")
            return None
        warm_copy = deepcopy(ent)
        warm_copy.live_id = uuid4()
        return warm_copy

    def list_by_type(self, entity_type: Type[Entity]) -> List[Entity]:
        return [
            deepcopy(e)
            for e in self._registry.values()
            if isinstance(e, entity_type)
        ]

    def get_many(self, entity_ids: List[UUID], expected_type: Optional[Type[Entity]]) -> List[Entity]:
        out: List[Entity] = []
        for eid in entity_ids:
            g = self.get(eid, expected_type)
            if g is not None:
                out.append(g)
        return out

    def get_registry_status(self) -> Dict[str, Any]:
        return {
            "in_memory": True,
            "entity_count": len(self._registry),
            "lineage_count": len(self._lineages),
        }

    def set_inference_orchestrator(self, orchestrator: object) -> None:
        self._inference_orchestrator = orchestrator

    def get_inference_orchestrator(self) -> Optional[object]:
        return self._inference_orchestrator

    def clear(self) -> None:
        self._registry.clear()
        self._entity_class_map.clear()
        self._lineages.clear()

    def get_lineage_entities(self, lineage_id: UUID) -> List[Entity]:
        """
        Return all entities in memory that share this lineage_id.
        """
        return [e for e in self._registry.values() if e.lineage_id == lineage_id]

    def has_lineage_id(self, lineage_id: UUID) -> bool:
        return any(e for e in self._registry.values() if e.lineage_id == lineage_id)

    def get_lineage_ids(self, lineage_id: UUID) -> List[UUID]:
        return [e.id for e in self._registry.values() if e.lineage_id == lineage_id]

    ############################
    # Helpers
    ############################
    def _store_cold_snapshot(self, e: Entity) -> None:
        snap = deepcopy(e)
        self._registry[e.id] = snap

        if e.lineage_id not in self._lineages:
            self._lineages[e.lineage_id] = []
        if e.id not in self._lineages[e.lineage_id]:
            self._lineages[e.lineage_id].append(e.id)


##############################
# 8) BaseEntitySQL (fallback)
##############################
try:
    # If you are using sqlmodel/SQLAlchemy, import them. 
    # We'll place them behind a try so this file can still run if not installed.
    from sqlmodel import SQLModel, Field, select, Session, Column, JSON
    from datetime import timezone

    def dynamic_import(path_str: str) -> Type[Entity]:
        """
        Very naive dynamic import, e.g. 'myapp.module.EntitySubclass'.
        Implementation is up to you. 
        For example:

            mod_name, cls_name = path_str.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            return getattr(mod, cls_name)

        Here, we just raise an error or return the base Entity as a fallback.
        """
        raise NotImplementedError("You must define a real dynamic_import function or use your own approach.")

    class BaseEntitySQL(SQLModel, table=True):  # type: ignore
        """
        Fallback table for storing *any* Entity if no specialized table is found.
        """
        id: UUID = Field(default_factory=uuid4, primary_key=True)
        lineage_id: UUID = Field(default_factory=uuid4, index=True)
        parent_id: Optional[UUID] = Field(default=None)
        created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
        old_ids: List[UUID] = Field(default_factory=list, sa_column=Column(JSON))  # type: ignore

        # The dotted Python path for the real class
        class_name: str = Field(...)

        # Non-versioning fields are stored here as JSON
        data: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))  # type: ignore

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
        def from_entity(cls, ent: Entity) -> 'BaseEntitySQL':  # Use string literal type
            versioning_fields = {"id", "lineage_id", "parent_id", "created_at", "old_ids", "live_id"}
            raw = ent.model_dump()  # pydantic's dictionary
            data_only = {k: v for k, v in raw.items() if k not in versioning_fields}
            return cls(
                id=ent.id,
                lineage_id=ent.lineage_id,
                parent_id=ent.parent_id,
                created_at=ent.created_at,
                old_ids=ent.old_ids,
                class_name=f"{ent.__class__.__module__}.{ent.__class__.__qualname__}",
                data=data_only
            )

except ImportError:
    # No SQLModel installed, so we skip this fallback definition
    BaseEntitySQL = None  # type: ignore
    SQLModel = None  # type: ignore
    Column = None  # type: ignore
    JSON = None  # type: ignore
    Session = None  # type: ignore
    select = None  # type: ignore


##############################
# 9) SqlEntityStorage
##############################
class SqlEntityStorage(EntityStorage):
    """
    Revised SQL-based storage with a fallback BaseEntitySQL table
    and a dictionary that maps UUID -> Python class to avoid scanning.
    """
    def __init__(
        self,
        session_factory: Callable[..., Any],
        entity_to_orm_map: Dict[Type[Entity], Type[Any]]
    ) -> None:
        self._logger = logging.getLogger("SqlEntityStorage")
        self._session_factory = session_factory
        self._entity_to_orm_map = entity_to_orm_map
        self._inference_orchestrator: Optional[object] = None

        # If using the fallback BaseEntitySQL, ensure Entity->BaseEntitySQL is in the map
        if BaseEntitySQL and (Entity not in self._entity_to_orm_map):
            self._entity_to_orm_map[Entity] = BaseEntitySQL

        # Keep a local dictionary: entity_id -> actual Python class
        self._entity_class_map: Dict[UUID, Type[Entity]] = {}

    def has_entity(self, entity_id: UUID) -> bool:
        if entity_id in self._entity_class_map:
            return True

        with self._session_factory() as sess:
            for ormcls in self._entity_to_orm_map.values():
                row = sess.get(ormcls, entity_id)
                if row is not None:
                    ent = row.to_entity()
                    self._entity_class_map[entity_id] = type(ent)
                    return True
        return False

    def get_cold_snapshot(self, entity_id: UUID) -> Optional[Entity]:
        cls_maybe = self._entity_class_map.get(entity_id)

        with self._session_factory() as sess:
            if cls_maybe and cls_maybe in self._entity_to_orm_map:
                row = sess.get(self._entity_to_orm_map[cls_maybe], entity_id)
                if row is not None:
                    return row.to_entity()
                return None
            # fallback scan if we don't know the class
            for ormcls in self._entity_to_orm_map.values():
                row = sess.get(ormcls, entity_id)
                if row is not None:
                    ent = row.to_entity()
                    self._entity_class_map[entity_id] = type(ent)
                    return ent
        return None

    def register(self, entity_or_id: Union[Entity, UUID]) -> Optional[Entity]:
        """Register entity with proper relationship handling."""
        if isinstance(entity_or_id, UUID):
            return self.get(entity_or_id, None)
            
        entity = entity_or_id
        
        # Log details to diagnose issues
        self._logger.info(f"Registering {type(entity).__name__}({entity.id}) with from_storage={getattr(entity, 'from_storage', False)}")
        
        # If from_storage, no need to check modifications
        if getattr(entity, 'from_storage', False):
            self._logger.debug(f"Entity {entity.id} is from storage, skipping modification check")
            return entity
                    
        old = self.get_cold_snapshot(entity.id)
        if old:
            # Use our structured comparison with logging
            self._logger.debug(f"Comparing {entity.id} with existing version")
            is_equal, diff_info = compare_values(entity, old)
            if not is_equal:
                self._logger.info(f"Changes detected in {entity.id}, forking with diff: {diff_info}")
                # Fork with diff info for proper nested handling
                entity = entity.fork(diff_info=diff_info)
                # Note: no recursive call to register - the fork already registers itself
            else:
                self._logger.debug(f"No changes detected in {entity.id}")
                
        # Store entity in DB
        ormcls = self._resolve_orm_cls(entity)
        if ormcls is None:
            self._logger.error(f"No ORM mapping found for {type(entity)}. Possibly add {type(entity)} -> BaseEntitySQL?")
            return None

        with self._session_factory() as sess:
            # Convert to SQL model
            sql_model = ormcls.from_entity(entity)
            
            # Save the SQL model
            sess.merge(sql_model)
            sess.flush()
            
            # Handle M2M relationships if present
            if hasattr(sql_model, 'sync_message_relationships') and hasattr(entity, 'history'):
                # Get fresh copy with relationships loaded
                db_model = sess.get(ormcls, sql_model.id)
                if db_model:
                    db_model = ormcls.sync_message_relationships(db_model, entity, sess)
                    
            sess.commit()

        self._entity_class_map[entity.id] = type(entity)
        return entity

    def get(self, entity_id: UUID, expected_type: Optional[Type[Entity]]) -> Optional[Entity]:
        cold = self.get_cold_snapshot(entity_id)
        if not cold:
            return None
        if expected_type and not isinstance(cold, expected_type):
            self._logger.error(f"Type mismatch: got {type(cold).__name__}, expected {expected_type.__name__}")
            return None
        warm_copy = deepcopy(cold)
        warm_copy.live_id = uuid4()
        return warm_copy

    def list_by_type(self, entity_type: Type[Entity]) -> List[Entity]:
        ormcls = self._entity_to_orm_map.get(entity_type)
        if not ormcls:
            # possibly fallback to BaseEntitySQL if entity_type is Entity
            if entity_type is Entity and BaseEntitySQL:
                ormcls = BaseEntitySQL
            else:
                return []

        out: List[Entity] = []
        if not (Session and select):
            self._logger.warning("SQLModel or SQLAlchemy not installed, cannot list_by_type.")
            return out

        with self._session_factory() as sess:
            rows = sess.exec(select(ormcls)).all()  # type: ignore
            for row in rows:
                ent = row.to_entity()
                self._entity_class_map[ent.id] = type(ent)
                out.append(ent)
        return out

    def get_many(self, entity_ids: List[UUID], expected_type: Optional[Type[Entity]]) -> List[Entity]:
        results: List[Entity] = []
        for eid in entity_ids:
            got = self.get(eid, expected_type)
            if got is not None:
                results.append(got)
        return results

    def get_registry_status(self) -> Dict[str, Any]:
        return {
            "storage": "sql",
            "known_ids_in_class_map": len(self._entity_class_map)
        }

    def set_inference_orchestrator(self, orchestrator: object) -> None:
        self._inference_orchestrator = orchestrator

    def get_inference_orchestrator(self) -> Optional[object]:
        return self._inference_orchestrator

    def clear(self) -> None:
        self._entity_class_map.clear()
        self._logger.warning("SqlEntityStorage.clear() not fully implemented - "
                             "you must manually truncate tables in the DB.")

    def _resolve_orm_cls(self, ent: Entity) -> Optional[Type[Any]]:
        # exact match
        cls_mapped = self._entity_to_orm_map.get(ent.__class__)
        if cls_mapped:
            return cls_mapped
        # fallback
        if Entity in self._entity_to_orm_map:
            return self._entity_to_orm_map[Entity]  # e.g. BaseEntitySQL
        return None

    ############################
    # Unified lineage approach
    ############################
    def get_lineage_entities(self, lineage_id: UUID) -> List[Entity]:
        if not (Session and select):
            self._logger.warning("SQLModel not installed, cannot get lineage entities.")
            return []
        out: List[Entity] = []
        with self._session_factory() as sess:
            for ormcls in self._entity_to_orm_map.values():
                stmt = select(ormcls).where(ormcls.lineage_id == lineage_id)
                rows = sess.exec(stmt).all()  # type: ignore
                for row in rows:
                    ent = row.to_entity()
                    self._entity_class_map[ent.id] = type(ent)
                    out.append(ent)
        return out

    def has_lineage_id(self, lineage_id: UUID) -> bool:
        ents = self.get_lineage_entities(lineage_id)
        return len(ents) > 0

    def get_lineage_ids(self, lineage_id: UUID) -> List[UUID]:
        return [x.id for x in self.get_lineage_entities(lineage_id)]


##############################
# 10) EntityRegistry (Facade)
##############################
class EntityRegistry(BaseRegistry):
    """
    A static registry class delegating to _storage for read/write ops.
    Unifies lineage building in a single place.
    """
    _logger = logging.getLogger("EntityRegistry")
    _storage: EntityStorage = InMemoryEntityStorage()  # default

    @classmethod
    def use_storage(cls, storage: EntityStorage) -> None:
        cls._storage = storage
        cls._logger.info(f"EntityRegistry now uses {type(storage).__name__}")

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
        base = super().get_registry_status()
        store_status = cls._storage.get_registry_status()
        return {**base, **store_status}

    @classmethod
    def set_inference_orchestrator(cls, orchestrator: object) -> None:
        cls._storage.set_inference_orchestrator(orchestrator)

    @classmethod
    def get_inference_orchestrator(cls) -> Optional[object]:
        return cls._storage.get_inference_orchestrator()

    @classmethod
    def clear(cls) -> None:
        cls._storage.clear()

    ##############
    # Unified lineage logic
    ##############
    @classmethod
    def has_lineage_id(cls, lineage_id: UUID) -> bool:
        return cls._storage.has_lineage_id(lineage_id)

    @classmethod
    def get_lineage_ids(cls, lineage_id: UUID) -> List[UUID]:
        return cls._storage.get_lineage_ids(lineage_id)

    @classmethod
    def get_lineage_entities(cls, lineage_id: UUID) -> List[Entity]:
        """Get all entities sharing a lineage_id."""
        return cls._storage.get_lineage_entities(lineage_id)

    @classmethod
    def build_lineage_tree(cls, lineage_id: UUID) -> Dict[UUID, Dict[str, Any]]:
        """
        Build a lineage tree from a list of Entities in that lineage.
        """
        nodes = cls._storage.get_lineage_entities(lineage_id)
        if not nodes:
            return {}

        by_id = {e.id: e for e in nodes}
        # find root(s)
        roots = [x for x in nodes if x.parent_id not in by_id]

        tree: Dict[UUID, Dict[str, Any]] = {}

        def build_sub(e: Entity, depth: int=0):
            diff_from_parent = None
            if e.parent_id and e.parent_id in by_id:
                parent = by_id[e.parent_id]
                # Use unified comparison for diff calculation
                is_equal, raw_diffs = compare_values(e, parent)
                if not is_equal:
                    # Convert to old format for compatibility
                    diff = EntityDiff()
                    for path, diff_info in raw_diffs.items():
                        field = path.split('.')[0] if '.' in path else path
                        field = field.split('[')[0] if '[' in field else field
                        diff.add_diff(field, diff_info["type"], 
                                  old_value=diff_info.get("old"), 
                                  new_value=diff_info.get("new"))
                    diff_from_parent = diff.field_diffs

            tree[e.id] = {
                "entity": e,
                "children": [],
                "depth": depth,
                "parent_id": e.parent_id,
                "created_at": e.created_at,
                "data": e.entity_dump(),
                "diff_from_parent": diff_from_parent
            }

            if e.parent_id and e.parent_id in tree:
                tree[e.parent_id]["children"].append(e.id)

            # children
            for child in nodes:
                if child.parent_id == e.id:
                    build_sub(child, depth+1)

        for r in roots:
            build_sub(r, 0)
        return tree

    @classmethod
    def get_lineage_tree_sorted(cls, lineage_id: UUID) -> Dict[str, Any]:
        tr = cls.build_lineage_tree(lineage_id)
        if not tr:
            return {"nodes": {}, "edges": [], "root": None, "sorted_ids": [], "diffs": {}}
        items_sorted = sorted(tr.items(), key=lambda x: x[1]["created_at"])
        sorted_ids = [k for k, _ in items_sorted]
        edges = []
        diffs = {}
        for vid, node_data in tr.items():
            p = node_data["parent_id"]
            if p:
                edges.append((p, vid))
                if node_data["diff_from_parent"]:
                    diffs[vid] = node_data["diff_from_parent"]
        root_candidates = [k for k,v in tr.items() if not v["parent_id"]]
        root_id = root_candidates[0] if root_candidates else None
        return {
            "nodes": tr,
            "edges": edges,
            "root": root_id,
            "sorted_ids": sorted_ids,
            "diffs": diffs
        }

    @classmethod
    def get_lineage_mermaid(cls, lineage_id: UUID) -> str:
        data = cls.get_lineage_tree_sorted(lineage_id)
        if not data["nodes"]:
            return "```mermaid\ngraph TD\n  No data available\n```"

        lines = ["```mermaid", "graph TD"]

        def format_value(val: Any) -> str:
            """Format value with type-specific handling."""
            if val is None:
                return "null"
            elif isinstance(val, (int, float, bool, str, UUID)):
                s = str(val)
                return s[:15] + "..." if len(s) > 15 else s
            elif isinstance(val, (list, tuple)):
                return f"[{len(val)} items]"
            elif isinstance(val, dict):
                return f"{{{len(val)} keys}}"
            elif isinstance(val, Entity):
                return f"Entity({val.id})"
            else:
                return f"{type(val).__name__}"

        # Add nodes with detailed change info
        for node_id, node_data in data["nodes"].items():
            ent = node_data["entity"]
            clsname = type(ent).__name__
            short = str(node_id)[:8]
            
            if not node_data["parent_id"]:
                # Root node
                dd = list(node_data["data"].items())[:3]
                summary = "\\n".join(f"{k}={format_value(v)}" for k,v in dd)
                lines.append(f"  {node_id}[\"{clsname}\\n{short}\\n{summary}\"]")
            else:
                # Change node with detailed diff
                diff = data["diffs"].get(node_id, {})
                changes = len(diff)
                if changes > 0:
                    change_details = []
                    for field, info in diff.items():
                        t = info.get("type")
                        if t == "modified":
                            old_val = format_value(info.get("old"))
                            new_val = format_value(info.get("new"))
                            change_details.append(f"{field}: {old_val}→{new_val}")
                        elif t == "added":
                            change_details.append(f"+{field}: {format_value(info.get('new'))}")
                        elif t == "removed":
                            change_details.append(f"-{field}: {format_value(info.get('old'))}")
                    
                    # Limit details for readability
                    if len(change_details) > 3:
                        change_details = change_details[:3] + [f"...({len(diff)-3} more)"]
                    
                    change_summary = "\\n".join(change_details)
                    lines.append(f"  {node_id}[\"{clsname}\\n{short}\\n{change_summary}\"]")
                else:
                    lines.append(f"  {node_id}[\"{clsname}\\n{short}\\n(No changes detected)\"]")

        # Edge labels with detailed changes
        for (p, c) in data["edges"]:
            diff = data["diffs"].get(c, {})
            if diff:
                label_parts = []
                for field, info in diff.items():
                    t = info.get("type")
                    if t == "modified":
                        old_val = format_value(info.get("old"))
                        new_val = format_value(info.get("new"))
                        label_parts.append(f"{field}: {old_val}→{new_val}")
                    elif t == "added":
                        label_parts.append(f"+{field}")
                    elif t == "removed":
                        label_parts.append(f"-{field}")
                
                if len(label_parts) > 3:
                    label_parts = label_parts[:3] + [f"...({len(diff)-3} more)"]
                label = "\\n".join(label_parts)
                lines.append(f"  {p} -->|\"{label}\"| {c}")
            else:
                lines.append(f"  {p} --> {c}")

        lines.append("```")
        return "\n".join(lines)


##############################
# 11) Enhanced Entity Tracer
##############################
def entity_tracer(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that checks/forks Entities for modifications before/after the function call.
    Uses dependency tracking for proper fork order.
    """
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Collect entities and their dependencies
            ents_with_deps = _collect_entities_with_dependencies(args, kwargs)
            
            # Fork bottom-up before function call
            _check_and_fork_with_dependencies(ents_with_deps)
            
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Recollect entities and their dependencies after function call
            ents_with_deps = _collect_entities_with_dependencies(args, kwargs)
            
            # Fork bottom-up after function call
            _check_and_fork_with_dependencies(ents_with_deps)
            
            # If result is a modified entity, return the appropriate version
            if isinstance(result, Entity) and id(result) in ents_with_deps:
                return ents_with_deps[id(result)][0]
            return result
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Collect entities and their dependencies
            ents_with_deps = _collect_entities_with_dependencies(args, kwargs)
            
            # Fork bottom-up before function call
            _check_and_fork_with_dependencies(ents_with_deps)
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Recollect entities and their dependencies after function call
            ents_with_deps = _collect_entities_with_dependencies(args, kwargs)
            
            # Fork bottom-up after function call
            _check_and_fork_with_dependencies(ents_with_deps)
            
            # If result is a modified entity, return the appropriate version
            if isinstance(result, Entity) and id(result) in ents_with_deps:
                return ents_with_deps[id(result)][0]
            return result
        return sync_wrapper