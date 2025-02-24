############################################################
# entity.py
############################################################

import json
import inspect
import logging
from typing import (
    Any, Dict, Optional, Type, TypeVar, List, Protocol, runtime_checkable,
    Union, Callable, get_args, cast, Self
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
# 2) The Entity + Diff
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

    NOTE: This is the original definition from your code, unchanged.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    live_id: UUID = Field(default_factory=uuid4, description="Live/warm identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    parent_id: Optional[UUID] = None
    lineage_id: UUID = Field(default_factory=uuid4)
    old_ids: List[UUID] = Field(default_factory=list)
    from_storage: bool = Field(default=False, description="Whether the entity was loaded from storage when loaded from storage as acold object we do not re-register the entity")
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda dt: dt.isoformat()
        }

    def register_entity(self: "Entity") -> "Entity":
        if not EntityRegistry.has_entity(self.id):
            EntityRegistry.register(self)
        elif not self.from_storage:
            cold = EntityRegistry.get_cold_snapshot(self.id)
            if cold and self.has_modifications(cold):
                self.fork()
        return self

    @model_validator(mode='after')
    def _auto_register(self) -> Self:
        entity = self.register_entity()
        return self

    def fork(self, force: bool = False, **kwargs: Any) -> "Entity":
        from __main__ import EntityRegistry
        cold = EntityRegistry.get_cold_snapshot(self.id)
        if cold is None:
            EntityRegistry.register(self)
            return self

        changed = False
        if kwargs:
            changed = True
            for k, v in kwargs.items():
                setattr(self, k, v)

        if not force and not changed and not self.has_modifications(cold):
            return self

        old_id = self.id
        self.id = uuid4()
        self.parent_id = old_id
        self.old_ids.append(old_id)
        EntityRegistry.register(self)
        return self

    def has_modifications(self, other: "Entity") -> bool:
        from __main__ import EntityRegistry
        EntityRegistry._logger.debug(f"Checking modifications between {self.id} and {other.id}")

        exclude = {'id', 'created_at', 'parent_id', 'live_id', 'old_ids', 'lineage_id'}
        self_fields = set(self.model_fields.keys()) - exclude
        other_fields = set(other.model_fields.keys()) - exclude
        if self_fields != other_fields:
            return True

        for f in self_fields:
            val_self = getattr(self, f)
            val_other = getattr(other, f)
            if val_self != val_other:
                return True
        return False

    def compute_diff(self, other: "Entity") -> EntityDiff:
        from __main__ import EntityRegistry
        EntityRegistry._logger.debug(f"Computing diff: {self.id} vs {other.id}")
        diff = EntityDiff()
        exclude = {'id','created_at','parent_id','live_id','old_ids','lineage_id'}
        sfields = set(self.model_fields.keys()) - exclude
        ofields = set(other.model_fields.keys()) - exclude

        for f in sfields - ofields:
            diff.add_diff(f, "added", new_value=getattr(self, f))
        for f in ofields - sfields:
            diff.add_diff(f, "removed", old_value=getattr(other, f))

        for f in sfields & ofields:
            sv = getattr(self, f)
            ov = getattr(other, f)
            if sv != ov:
                diff.add_diff(f, "modified", old_value=ov, new_value=sv)
        return diff

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

    @classmethod
    def compare_entities(cls, e1: "Entity", snapshot: Dict[str, Any]) -> bool:
        data1 = e1.entity_dump()
        for k in ['id','created_at','parent_id','live_id','old_ids','lineage_id']:
            data1.pop(k, None)
        keys = set(data1.keys()) | set(snapshot.keys())
        for k in keys:
            if data1.get(k) != snapshot.get(k):
                return False
        return True


##############################
# 3) EntityStorage Protocol
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
    # New method to unify lineage logic externally
    def get_lineage_entities(self, lineage_id: UUID) -> List[Entity]: ...
    def has_lineage_id(self, lineage_id: UUID) -> bool: ...
    def get_lineage_ids(self, lineage_id: UUID) -> List[UUID]: ...


##############################
# 4) InMemoryEntityStorage
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
        if isinstance(entity_or_id, UUID):
            return self.get(entity_or_id, None)

        e = entity_or_id
        old = self._registry.get(e.id)
        if not old:
            self._store_cold_snapshot(e)
            return e

        if e.has_modifications(old):
            e.fork(force=True)
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
# 5) BaseEntitySQL (fallback)
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
# 6) SqlEntityStorage
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
        if isinstance(entity_or_id, UUID):
            return self.get(entity_or_id, None)

        ent = entity_or_id
        old = self.get_cold_snapshot(ent.id)
        if old and ent.has_modifications(old):
            ent.fork(force=True)
            return self.register(ent)  # re-register with the new ID

        ormcls = self._resolve_orm_cls(ent)
        if ormcls is None:
            self._logger.error(f"No ORM mapping found for {type(ent)}. Possibly add {type(ent)} -> BaseEntitySQL?")
            return None

        row = ormcls.from_entity(ent)
        with self._session_factory() as sess:
            sess.merge(row)
            sess.commit()

        self._entity_class_map[ent.id] = type(ent)
        return ent

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
# 7) EntityRegistry (Facade)
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
                diff = e.compute_diff(parent)
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
            s = str(val)
            return s[:15] + "..." if len(s) > 15 else s

        for node_id, node_data in data["nodes"].items():
            ent = node_data["entity"]
            clsname = type(ent).__name__
            short = str(node_id)[:8]
            if not node_data["parent_id"]:
                # root
                dd = list(node_data["data"].items())[:3]
                summary = "\\n".join(f"{k}={format_value(v)}" for k,v in dd)
                lines.append(f"  {node_id}[\"{clsname}\\n{short}\\n{summary}\"]")
            else:
                diff = data["diffs"].get(node_id, {})
                changes = len(diff)
                lines.append(f"  {node_id}[\"{clsname}\\n{short}\\n({changes} changes)\"]")

        for (p, c) in data["edges"]:
            diff = data["diffs"].get(c, {})
            if diff:
                label_parts = []
                for field, info in diff.items():
                    t = info.get("type")
                    if t == "modified":
                        label_parts.append(f"{field} mod")
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
# 8) Decorators for versioning
##############################
def _collect_entities(args: tuple, kwargs: dict) -> Dict[int, Entity]:
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
    from __main__ import EntityRegistry
    cold = EntityRegistry.get_cold_snapshot(entity.id)
    if cold and entity.has_modifications(cold):
        entity.fork()

def entity_tracer(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that checks/forks Entities for modifications before/after the function call.
    """
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            ents_before = _collect_entities(args, kwargs)
            for e in ents_before.values():
                _check_and_fork_modified(e)
            result = await func(*args, **kwargs)
            for e in ents_before.values():
                _check_and_fork_modified(e)
            if isinstance(result, Entity) and id(result) in ents_before:
                return ents_before[id(result)]
            return result
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            ents_before = _collect_entities(args, kwargs)
            for e in ents_before.values():
                _check_and_fork_modified(e)
            result = func(*args, **kwargs)
            for e in ents_before.values():
                _check_and_fork_modified(e)
            if isinstance(result, Entity) and id(result) in ents_before:
                return ents_before[id(result)]
            return result
        return sync_wrapper
