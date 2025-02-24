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

# We keep this import from your code. If you do not need it, remove it.
from minference.ecs.base_registry import BaseRegistry


###############################################################################
# 1) Protocol + Generics
###############################################################################

@runtime_checkable
class HasID(Protocol):
    """Protocol requiring an `id: UUID` field."""
    id: UUID

###############################################################################
# 2) EntityDiff class
###############################################################################

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

###############################################################################
# 3) The Entity base class
###############################################################################

class Entity(BaseModel):
    """
    Base class for registry-integrated, serializable entities.
    Subclasses are responsible for custom serialization logic,
    possibly nested relationships, etc.
    Snapshots + re-registration => auto-versioning if fields change in place.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this entity instance")
    live_id: UUID = Field(default_factory=uuid4, description="Stable identifier for warm/active instance that persists across forks")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when entity was created")
    parent_id: Optional[UUID] = None
    lineage_id: UUID = Field(default_factory=uuid4)
    old_ids: List[UUID] = Field(default_factory=list)

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda dt: dt.isoformat()
        }

    ############################################################################
    # A normal method, not a pydantic validator
    ############################################################################
    def register_entity(self: "Entity") -> "Entity":
        """
        Retro-compatible method that manually registers (or forks) this entity.
        """
        from __main__ import EntityRegistry
        if not EntityRegistry.has_entity(self.id):
            EntityRegistry.register(self)
        else:
            cold = EntityRegistry.get_cold_snapshot(self.id)
            if cold and self.has_modifications(cold):
                self.fork()
        return self

    ############################################################################
    # Pydantic v2 after-model validator
    ############################################################################
    @model_validator(mode='after')
    def _auto_register(self) -> Self:
        """
        If a user creates an entity instance, this ensures it is registered
        automatically at the end of pydantic validation (like the old code).
        """
        if not EntityRegistry.has_entity(self.id):
            EntityRegistry.register(self)
        else:
            cold = EntityRegistry.get_cold_snapshot(self.id)
            if cold and self.has_modifications(cold):
                self.fork()
        return self

    ############################################################################
    # forking logic
    ############################################################################
    def fork(self, force: bool = False, **kwargs: Any) -> "Entity":
        """
        If modifications are detected or force=True, produce a new version
        with a new `id`, setting `parent_id` to the old `id`, and storing
        the old id in `old_ids`.
        """
        from __main__ import EntityRegistry
        cold = EntityRegistry.get_cold_snapshot(self.id)
        if cold is None:
            EntityRegistry.register(self)
            return self

        # apply modifications from kwargs
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

    ############################################################################
    # has_modifications
    ############################################################################
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

    ############################################################################
    # compute_diff
    ############################################################################
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

    ############################################################################
    # entity_dump
    ############################################################################
    def entity_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Similar to .model_dump but we skip the versioning fields, plus
        recursively dumps nested Entities.
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

    ############################################################################
    # Convenience getters (list_all, get_many, etc.)
    ############################################################################
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

    ############################################################################
    # compare_entities
    ############################################################################
    @classmethod
    def compare_entities(cls, e1: "Entity", snapshot: Dict[str, Any]) -> bool:
        data1 = e1.entity_dump()
        # remove versioning keys
        for k in ['id','created_at','parent_id','live_id','old_ids','lineage_id']:
            data1.pop(k, None)
        keys = set(data1.keys()) | set(snapshot.keys())
        for k in keys:
            if data1.get(k) != snapshot.get(k):
                return False
        return True


###############################################################################
# 4) The Storage Protocol
###############################################################################

class EntityStorage(Protocol):
    """
    Generic interface for storing and retrieving Entities, building lineage, etc.
    We do NOT parametrize it with a TypeVar because Pyright complains
    that \"Expected no type arguments\" if we tried EntityStorage[T].
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
    def has_lineage_id(self, lineage_id: UUID) -> bool: ...
    def get_lineage_ids(self, lineage_id: UUID) -> List[UUID]: ...
    def build_lineage_tree(self, lineage_id: UUID) -> Dict[UUID, Dict[str,Any]]: ...
    def get_lineage_tree_sorted(self, lineage_id: UUID) -> Dict[str, Any]: ...
    def get_lineage_mermaid(self, lineage_id: UUID) -> str: ...
    def clear(self) -> None: ...


###############################################################################
# 5) The InMemoryEntityStorage
###############################################################################

class InMemoryEntityStorage(EntityStorage):
    """
    Default in-memory storage that mirrors the original dictionary-based approach.
    Perfectly preserves the old logic for retro-compatibility.
    """
    def __init__(self) -> None:
        self._logger = logging.getLogger("InMemoryEntityStorage")
        self._registry: Dict[UUID, Entity] = {}
        self._timestamps: Dict[UUID, datetime] = {}
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
        if e.id not in self._registry:
            self._store_cold_snapshot(e)
            return e

        cold = self._registry[e.id]
        if e.has_modifications(cold):
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
        return [deepcopy(e) for e in self._registry.values() if isinstance(e, entity_type)]

    def get_many(self, entity_ids: List[UUID], expected_type: Optional[Type[Entity]]) -> List[Entity]:
        results: List[Entity] = []
        for uid in entity_ids:
            got = self.get(uid, expected_type)
            if got is not None:
                results.append(got)
        return results

    def get_registry_status(self) -> Dict[str, Any]:
        type_counts: Dict[str,int] = {}
        for e in self._registry.values():
            nm = e.__class__.__name__
            type_counts[nm] = type_counts.get(nm, 0) + 1
        timestamps = sorted(self._timestamps.values())
        total_lineages = len(self._lineages)
        total_versions = sum(len(v) for v in self._lineages.values())
        return {
            "in_memory": True,
            "entities_by_type": type_counts,
            "timestamps_count": len(timestamps),
            "lineage_count": total_lineages,
            "version_count": total_versions
        }

    def set_inference_orchestrator(self, orchestrator: object) -> None:
        self._inference_orchestrator = orchestrator

    def get_inference_orchestrator(self) -> Optional[object]:
        return self._inference_orchestrator

    def has_lineage_id(self, lineage_id: UUID) -> bool:
        return lineage_id in self._lineages

    def get_lineage_ids(self, lineage_id: UUID) -> List[UUID]:
        return self._lineages.get(lineage_id, [])

    def build_lineage_tree(self, lineage_id: UUID) -> Dict[UUID, Dict[str, Any]]:
        version_ids = self._lineages.get(lineage_id, [])
        if not version_ids:
            self._logger.info(f"No versions found for lineage {lineage_id}")
            return {}
        # find root
        root_id: Optional[UUID] = None
        for vid in version_ids:
            ent = self._registry.get(vid)
            if ent and ent.parent_id is None:
                root_id = vid
                break
        if not root_id:
            self._logger.error(f"No root node found for lineage {lineage_id}")
            return {}

        tree: Dict[UUID,Dict[str,Any]] = {}

        def build_sub(node_id: UUID, depth:int=0) -> None:
            ent = self._registry.get(node_id)
            if not ent:
                return
            diff_from_parent = None
            if ent.parent_id:
                parent = self._registry.get(ent.parent_id)
                if parent:
                    diff = ent.compute_diff(parent)
                    diff_from_parent = diff.field_diffs

            tree[node_id] = {
                "entity": ent,
                "children": [],
                "depth": depth,
                "parent_id": ent.parent_id,
                "created_at": ent.created_at,
                "data": ent.entity_dump(),
                "diff_from_parent": diff_from_parent
            }
            if ent.parent_id and ent.parent_id in tree:
                tree[ent.parent_id]["children"].append(node_id)

            # find children
            for cvid in version_ids:
                if cvid == node_id:
                    continue
                c = self._registry.get(cvid)
                if c and c.parent_id == node_id:
                    build_sub(cvid, depth+1)

        build_sub(root_id)
        return tree

    def get_lineage_tree_sorted(self, lineage_id: UUID) -> Dict[str, Any]:
        tr = self.build_lineage_tree(lineage_id)
        if not tr:
            return {"nodes":{}, "edges":[], "root":None, "sorted_ids":[], "diffs":{}}
        # sort by created_at
        items_sorted = sorted(tr.items(), key=lambda x: x[1]["created_at"])
        sorted_ids = [kv[0] for kv in items_sorted]
        edges: List[tuple[UUID,UUID]] = []
        diffs: Dict[UUID,Any] = {}
        for vid,node_dat in tr.items():
            p = node_dat["parent_id"]
            if p:
                edges.append((p,vid))
                if node_dat["diff_from_parent"]:
                    diffs[vid] = node_dat["diff_from_parent"]
        # find root
        root_candidates = [vid for vid,nd in tr.items() if not nd["parent_id"]]
        root_id = root_candidates[0] if root_candidates else None
        return {
            "nodes": tr,
            "edges": edges,
            "root": root_id,
            "sorted_ids": sorted_ids,
            "diffs": diffs
        }

    def get_lineage_mermaid(self, lineage_id: UUID) -> str:
        """Generate a Mermaid graph visualization of the lineage tree."""
        data = self.get_lineage_tree_sorted(lineage_id)
        if not data["nodes"]:
            return "```mermaid\ngraph TD\n  No data available\n```"
            
        lines = ["```mermaid", "graph TD"]
        
        def format_value(value: Any) -> str:
            """Format a value for display in Mermaid."""
            if isinstance(value, (list, tuple)):
                return f"[{len(value)} items]"
            elif isinstance(value, dict):
                return f"{{{len(value)} keys}}"
            else:
                val_str = str(value)[:20]
                return val_str + "..." if len(str(value)) > 20 else val_str

        # Add nodes
        for node_id, node_data in data["nodes"].items():
            entity = node_data["entity"]
            node_type = type(entity).__name__
            short_id = str(node_id)[:8]
            
            # For root node, show initial state
            if not node_data["parent_id"]:
                data_summary = [
                    f"{key}={format_value(value)}"
                    for key, value in node_data["data"].items()
                    if not isinstance(value, (bytes, bytearray))
                ][:3]  # Limit to 3 fields
                if len(node_data["data"]) > 3:
                    data_summary.append(f"...({len(node_data['data'])-3} more)")
                data_text = "\\n" + "\\n".join(data_summary) if data_summary else ""
                lines.append(f"  {node_id}[\"{node_type}\\n{short_id}{data_text}\"]")
            else:
                # For non-root nodes, show type, ID, and modification count
                diff = node_data.get("diff_from_parent", {})
                mod_count = len(diff) if diff else 0
                lines.append(f"  {node_id}[\"{node_type}\\n{short_id}\\n({mod_count} changes)\"]")

        # Add edges with diff information
        for parent_id, child_id in data["edges"]:
            diff = data["diffs"].get(child_id, {})
            if not diff:
                lines.append(f"  {parent_id} --> {child_id}")
                continue
                
            # Format changes for edge label
            changes = []
            for field, diff_info in diff.items():
                diff_type = diff_info.get("type", "")
                if diff_type == "modified":
                    old_val = diff_info.get("old")
                    new_val = diff_info.get("new")
                    if isinstance(old_val, (dict, list, bytes, bytearray)) or \
                       isinstance(new_val, (dict, list, bytes, bytearray)):
                        changes.append(f"{field} updated")
                    else:
                        changes.append(f"{field}: {format_value(old_val)}→{format_value(new_val)}")
                elif diff_type == "added":
                    changes.append(f"+{field}")
                elif diff_type == "removed":
                    changes.append(f"-{field}")
                elif diff_type == "entity_modified":
                    changes.append(f"{field}* modified")
                elif diff_type == "list_modified":
                    old_len = diff_info.get("old", {}).get("length", 0)
                    new_len = diff_info.get("new", {}).get("length", 0)
                    changes.append(f"{field}[{old_len}→{new_len}]")
            
            if changes:
                # Limit changes shown to prevent long edge labels
                if len(changes) > 3:
                    changes = changes[:3] + [f"...({len(changes)-3} more)"]
                diff_text = "\\n".join(changes)
                lines.append(f"  {parent_id} -->|\"{diff_text}\"| {child_id}")
            else:
                lines.append(f"  {parent_id} --> {child_id}")

        lines.append("```")
        return "\n".join(lines)

    def clear(self) -> None:
        self._registry.clear()
        self._timestamps.clear()
        self._lineages.clear()

    def _store_cold_snapshot(self, e: Entity) -> None:
        snap = deepcopy(e)
        self._registry[e.id] = snap
        self._timestamps[e.id] = datetime.utcnow()
        if e.lineage_id not in self._lineages:
            self._lineages[e.lineage_id] = []
        if e.id not in self._lineages[e.lineage_id]:
            self._lineages[e.lineage_id].append(e.id)

###############################################################################
# 6) SqlEntityStorage
###############################################################################

class SqlEntityStorage(EntityStorage):
    """
    Example SQL-based storage that:
      1) Accepts a session_factory for producing sessions (SQLAlchemy or SQLModel).
      2) Accepts a dictionary mapping your domain `Entity` classes
         to corresponding ORM classes with `.from_entity(...) -> ORM` and `.to_entity() -> Entity`.
    Because we can't import your sql_models here, you pass them in from outside.
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

    def has_entity(self, entity_id: UUID) -> bool:
        with self._session_factory() as sess:
            for ormcls in self._entity_to_orm_map.values():
                row = sess.get(ormcls, entity_id)
                if row:
                    return True
        return False

    def get_cold_snapshot(self, entity_id: UUID) -> Optional[Entity]:
        with self._session_factory() as sess:
            for ormcls in self._entity_to_orm_map.values():
                row = sess.get(ormcls, entity_id)
                if row is not None:
                    return row.to_entity()
        return None

    def register(self, entity_or_id: Union[Entity, UUID]) -> Optional[Entity]:
        if isinstance(entity_or_id, UUID):
            return self.get(entity_or_id, None)
        ent = entity_or_id
        # check existing
        old = self.get_cold_snapshot(ent.id)
        if old and ent.has_modifications(old):
            ent.fork(force=True)
            return ent
        # normal upsert
        ormcls = self._resolve_orm_cls(ent)
        if not ormcls:
            self._logger.error(f"No ORM mapping found for {type(ent)}")
            return None
        row = ormcls.from_entity(ent)
        with self._session_factory() as sess:
            sess.merge(row)
            sess.commit()
        return ent

    def get(self, entity_id: UUID, expected_type: Optional[Type[Entity]]) -> Optional[Entity]:
        with self._session_factory() as sess:
            if expected_type and expected_type in self._entity_to_orm_map:
                row = sess.get(self._entity_to_orm_map[expected_type], entity_id)
                if row:
                    e = row.to_entity()
                    e.live_id = uuid4()
                    return e
                return None
            # naive approach
            for ormcls in self._entity_to_orm_map.values():
                row = sess.get(ormcls, entity_id)
                if row:
                    e = row.to_entity()
                    e.live_id = uuid4()
                    if expected_type and not isinstance(e, expected_type):
                        return None
                    return e
        return None

    def list_by_type(self, entity_type: Type[Entity]) -> List[Entity]:
        ormcls = self._entity_to_orm_map.get(entity_type)
        if not ormcls:
            return []
        with self._session_factory() as sess:
            rows = sess.exec(ormcls.select()).all()  # type: ignore
            out: List[Entity] = []
            for r in rows:
                e = r.to_entity()
                e.live_id = uuid4()
                out.append(e)
            return out

    def get_many(self, entity_ids: List[UUID], expected_type: Optional[Type[Entity]]) -> List[Entity]:
        results: List[Entity] = []
        leftover = set(entity_ids)
        with self._session_factory() as sess:
            if expected_type and expected_type in self._entity_to_orm_map:
                ormcls = self._entity_to_orm_map[expected_type]
                rows = sess.exec(ormcls.select().where(ormcls.id.in_(list(leftover)))).all()  # type: ignore
                for r in rows:
                    e = r.to_entity()
                    e.live_id = uuid4()
                    results.append(e)
                return results
            # naive
            for ormcls in self._entity_to_orm_map.values():
                if not leftover:
                    break
                rows = sess.exec(ormcls.select().where(ormcls.id.in_(list(leftover)))).all()  # type: ignore
                for r in rows:
                    e = r.to_entity()
                    e.live_id = uuid4()
                    results.append(e)
                    leftover.discard(e.id)
        return results

    def get_registry_status(self) -> Dict[str,Any]:
        return {
            "storage": "sql",
            "note": "No direct line counts here"
        }

    def set_inference_orchestrator(self, orchestrator: object) -> None:
        self._inference_orchestrator = orchestrator

    def get_inference_orchestrator(self) -> Optional[object]:
        return self._inference_orchestrator

    def has_lineage_id(self, lineage_id: UUID) -> bool:
        with self._session_factory() as sess:
            for ormcls in self._entity_to_orm_map.values():
                row = sess.exec(ormcls.select().where(ormcls.lineage_id==lineage_id)).first()  # type: ignore
                if row:
                    return True
        return False

    def get_lineage_ids(self, lineage_id: UUID) -> List[UUID]:
        ids: List[UUID] = []
        with self._session_factory() as sess:
            for ormcls in self._entity_to_orm_map.values():
                rows = sess.exec(ormcls.select().where(ormcls.lineage_id==lineage_id)).all()  # type: ignore
                for r in rows:
                    ids.append(r.id)
        return ids

    def build_lineage_tree(self, lineage_id: UUID) -> Dict[UUID,Dict[str,Any]]:
        with self._session_factory() as sess:
            nodes: Dict[UUID, Entity] = {}
            for ormcls in self._entity_to_orm_map.values():
                rows = sess.exec(ormcls.select().where(ormcls.lineage_id==lineage_id)).all()  # type: ignore
                for row in rows:
                    e = row.to_entity()
                    nodes[e.id] = e
        if not nodes:
            return {}
        # find root
        root_id = None
        for k,v in nodes.items():
            if v.parent_id is None:
                root_id = k
                break
        if not root_id:
            return {}
        tree: Dict[UUID,Dict[str,Any]] = {}
        def build_sub(nid: UUID, depth:int=0) -> None:
            ent = nodes[nid]
            diff = None
            if ent.parent_id and ent.parent_id in nodes:
                pass  # you can do compute_diff if you want
            tree[nid] = {
                "entity": ent,
                "children": [],
                "depth": depth,
                "parent_id": ent.parent_id,
                "created_at": ent.created_at,
                "data": ent.entity_dump(),
                "diff_from_parent": diff
            }
            if ent.parent_id and ent.parent_id in tree:
                tree[ent.parent_id]["children"].append(nid)
            # find children
            for kid, e2 in nodes.items():
                if e2.parent_id == nid and kid!=nid:
                    build_sub(kid, depth+1)
        build_sub(root_id)
        return tree

    def get_lineage_tree_sorted(self, lineage_id: UUID) -> Dict[str,Any]:
        tr = self.build_lineage_tree(lineage_id)
        if not tr:
            return {"nodes":{}, "edges":[], "root":None, "sorted_ids":[], "diffs":{}}
        items_sorted = sorted(tr.items(), key=lambda x: x[1]["created_at"])
        sorted_ids = [x[0] for x in items_sorted]
        edges: List[tuple[UUID,UUID]] = []
        diffs: Dict[UUID,Any] = {}
        for vid,nd in tr.items():
            p = nd["parent_id"]
            if p:
                edges.append((p,vid))
                if nd["diff_from_parent"]:
                    diffs[vid] = nd["diff_from_parent"]
        roots = [vid for vid, nd2 in tr.items() if not nd2["parent_id"]]
        root_id = roots[0] if roots else None
        return {
            "nodes": tr,
            "edges": edges,
            "root": root_id,
            "sorted_ids": sorted_ids,
            "diffs": diffs
        }

    def get_lineage_mermaid(self, lineage_id: UUID) -> str:
        data = self.get_lineage_tree_sorted(lineage_id)
        if not data["nodes"]:
            return "```mermaid\ngraph TD\n  No data available\n```"
        lines = ["```mermaid","graph TD"]
        lines.append("```")
        return "\n".join(lines)

    def clear(self) -> None:
        # Typically you'd do TRUNCATE in a real db, but let's just log
        self._logger.warning("SqlEntityStorage.clear() not implemented yet.")

    ###########################################################################
    # internal
    ###########################################################################
    def _resolve_orm_cls(self, ent: Entity) -> Optional[Type[Any]]:
        for ecls, ormcls in self._entity_to_orm_map.items():
            if isinstance(ent, ecls):
                return ormcls
        return None


###############################################################################
# 7) The final EntityRegistry facade
###############################################################################

class EntityRegistry(BaseRegistry):
    """
    A static registry class that calls `_storage` for all operations.
    Defaults to InMemoryEntityStorage but can be swapped for a SQL
    or other backend.
    """
    _logger = logging.getLogger("EntityRegistry")
    _storage: EntityStorage = InMemoryEntityStorage()

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
        base = super().get_registry_status()  # calls BaseRegistry's status
        store_status = cls._storage.get_registry_status()
        return {**base, **store_status}

    @classmethod
    def set_inference_orchestrator(cls, orchestrator: object) -> None:
        cls._storage.set_inference_orchestrator(orchestrator)

    @classmethod
    def get_inference_orchestrator(cls) -> Optional[object]:
        return cls._storage.get_inference_orchestrator()

    @classmethod
    def has_lineage_id(cls, lineage_id: UUID) -> bool:
        return cls._storage.has_lineage_id(lineage_id)

    @classmethod
    def get_lineage_ids(cls, lineage_id: UUID) -> List[UUID]:
        return cls._storage.get_lineage_ids(lineage_id)

    @classmethod
    def build_lineage_tree(cls, lineage_id: UUID) -> Dict[UUID, Dict[str, Any]]:
        return cls._storage.build_lineage_tree(lineage_id)

    @classmethod
    def get_lineage_tree_sorted(cls, lineage_id: UUID) -> Dict[str, Any]:
        return cls._storage.get_lineage_tree_sorted(lineage_id)

    @classmethod
    def get_lineage_mermaid(cls, lineage_id: UUID) -> str:
        return cls._storage.get_lineage_mermaid(lineage_id)

    @classmethod
    def clear(cls) -> None:
        cls._storage.clear()


###############################################################################
# 8) Decorators for versioning
###############################################################################

def _collect_entities(args: tuple, kwargs: dict) -> Dict[int, Entity]:
    """
    Recursively find all Entity objects in the arguments (including nested
    lists/dicts/tuples/sets). Return a dict id->Entity.
    """
    found: Dict[int,Entity] = {}
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
    Returns the same type as the original function, ignoring type issues.
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