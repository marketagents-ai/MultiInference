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
from copy import deepcopy

from pydantic import BaseModel, Field, model_validator

# This is your original base registry import
from minference.ecs.base_registry import BaseRegistry

########################################
# 1) Protocol + Generics
########################################

@runtime_checkable
class HasID(Protocol):
    """Protocol requiring an `id: UUID` field."""
    id: UUID

# T_Entity will represent an Entity or its subclass.
T_Entity = TypeVar('T_Entity', bound='Entity')
T_Self = TypeVar('T_Self', bound='Entity')

class EntityDiff:
    """Represents structured differences between entities"""
    def __init__(self):
        self.field_diffs: Dict[str, Dict[str, Any]] = {}
    
    def add_diff(self, field: str, diff_type: str, old_value: Any = None, new_value: Any = None):
        self.field_diffs[field] = {
            "type": diff_type,
            "old": old_value,
            "new": new_value
        }

########################################
# 2) The Entity class
########################################

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
    parent_id: Optional[UUID] = Field(default=None, description="If set, points to parent's version ID")
    lineage_id: UUID = Field(default_factory=uuid4, description="Stable ID for entire lineage of versions.")
    old_ids: List[UUID] = Field(default_factory=list, description="List of previous IDs for this entity")
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda dt: dt.isoformat()
        }

    def entity_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Custom serialization method for entities that handles nested entities and lists.
        Similar to model_dump but with special handling for entity comparisons.
        """
        exclude_keys = kwargs.get('exclude', set())
        exclude_keys = exclude_keys.union({'id', 'created_at', 'lineage_id', 'parent_id', 'live_id'})
        kwargs['exclude'] = exclude_keys
        
        # Get base dump
        data = self.model_dump(*args, **kwargs)
        
        # Convert any nested entities to their serialized form for comparison
        for key, value in data.items():
            if isinstance(value, list):
                # Handle lists of entities or mixed content
                data[key] = [
                    item.entity_dump(exclude=exclude_keys) if isinstance(item, Entity) else item 
                    for item in value
                ]
            elif isinstance(value, Entity):
                # Handle nested entities
                data[key] = value.entity_dump(exclude=exclude_keys)
                
        return data

    def has_modifications(self, other: 'Entity') -> bool:
        """Compare this entity with another entity directly."""
        EntityRegistry._logger.debug(f"Checking modifications between entities {self.id} and {other.id}")
        self_fields = set(self.model_fields.keys()) - {'id', 'created_at', 'parent_id'}
        other_fields = set(other.model_fields.keys()) - {'id', 'created_at', 'parent_id'}

        if self_fields != other_fields:
            EntityRegistry._logger.debug(f"Field set mismatch between entities {self.id} and {other.id}")
            return True

        for field_name in self_fields:
            self_value = getattr(self, field_name)
            other_value = getattr(other, field_name)

            if isinstance(self_value, Entity) and isinstance(other_value, Entity):
                if self_value.has_modifications(other_value):
                    EntityRegistry._logger.debug(f"Nested entity modification detected in field {field_name}")
                    return True
            elif isinstance(self_value, (list, tuple, set)) and isinstance(other_value, (list, tuple, set)):
                if len(self_value) != len(other_value):
                    EntityRegistry._logger.debug(f"List length mismatch in field {field_name}")
                    return True
                for self_item, other_item in zip(self_value, other_value):
                    if isinstance(self_item, Entity) and isinstance(other_item, Entity):
                        if self_item.has_modifications(other_item):
                            EntityRegistry._logger.debug(f"List item modification detected in field {field_name}")
                            return True
                    elif self_item != other_item:
                        EntityRegistry._logger.debug(f"List item value mismatch in field {field_name}")
                        return True
            elif self_value != other_value:
                EntityRegistry._logger.debug(f"Value mismatch in field {field_name}")
                return True

        EntityRegistry._logger.debug(f"No modifications detected between entities {self.id} and {other.id}")
        return False

    def compute_diff(self, other: 'Entity') -> EntityDiff:
        """Compute detailed differences between this entity and another."""
        EntityRegistry._logger.debug(f"Computing diff between entities {self.id} and {other.id}")
        diff = EntityDiff()
        
        self_fields = set(self.model_fields.keys()) - {'id', 'created_at', 'parent_id'}
        other_fields = set(other.model_fields.keys()) - {'id', 'created_at', 'parent_id'}
        
        # Find added/removed fields
        for field in self_fields - other_fields:
            EntityRegistry._logger.debug(f"Field {field} added")
            diff.add_diff(field, "added", new_value=getattr(self, field))
        for field in other_fields - self_fields:
            EntityRegistry._logger.debug(f"Field {field} removed")
            diff.add_diff(field, "removed", old_value=getattr(other, field))
        
        # Compare common fields
        for field in self_fields & other_fields:
            self_value = getattr(self, field)
            other_value = getattr(other, field)
            
            if isinstance(self_value, Entity) and isinstance(other_value, Entity):
                nested_diff = self_value.compute_diff(other_value)
                if nested_diff.field_diffs:
                    EntityRegistry._logger.debug(f"Nested entity modifications in field {field}")
                    diff.add_diff(field, "entity_modified", 
                                old_value=other_value.id,
                                new_value={"entity_id": self_value.id, "changes": nested_diff.field_diffs})
            
            elif isinstance(self_value, (list, tuple, set)) and isinstance(other_value, (list, tuple, set)):
                if len(self_value) != len(other_value):
                    EntityRegistry._logger.debug(f"List length change in field {field}")
                    diff.add_diff(field, "list_modified",
                                old_value={"length": len(other_value), "items": other_value},
                                new_value={"length": len(self_value), "items": self_value})
                else:
                    list_changes = []
                    for idx, (self_item, other_item) in enumerate(zip(self_value, other_value)):
                        if isinstance(self_item, Entity) and isinstance(other_item, Entity):
                            item_diff = self_item.compute_diff(other_item)
                            if item_diff.field_diffs:
                                EntityRegistry._logger.debug(f"List item modification at index {idx} in field {field}")
                                list_changes.append({
                                    "index": idx,
                                    "type": "entity_modified",
                                    "changes": item_diff.field_diffs
                                })
                        elif self_item != other_item:
                            EntityRegistry._logger.debug(f"List item value change at index {idx} in field {field}")
                            list_changes.append({
                                "index": idx,
                                "type": "modified",
                                "old": other_item,
                                "new": self_item
                            })
                    if list_changes:
                        diff.add_diff(field, "list_modified", 
                                    old_value={"length": len(other_value)},
                                    new_value={"length": len(self_value), "changes": list_changes})
            
            elif self_value != other_value:
                EntityRegistry._logger.debug(f"Field {field} modified")
                diff.add_diff(field, "modified", old_value=other_value, new_value=self_value)
        
        return diff
    
    def _apply_modifications_and_create_version(self, cold_snapshot: 'Entity', force: bool, **kwargs) -> bool:
        """Helper method to apply modifications and create a new version if needed.
        
        Args:
            cold_snapshot: The cold snapshot to compare against
            force: Whether to force a new version
            **kwargs: Modifications to apply
            
        Returns:
            bool: True if a new version was created, False otherwise
        """
        # Apply kwargs modifications if any.
        if kwargs:
            EntityRegistry._logger.debug(f"Applying modifications to entity {self.id}: {kwargs}")
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        # Check for modifications (this covers both external changes and kwargs)
        if not force and not self.has_modifications(cold_snapshot):
            EntityRegistry._logger.debug(f"No actual modifications and not forced - returning entity {self.id}")
            return False
            
        # Create new version since modifications exist.
        old_id = self.id
        self.id = uuid4()
        self.old_ids.append(old_id)
        self.parent_id = old_id
        EntityRegistry._logger.debug(f"Creating new version: {old_id} -> {self.id}")
        return True

    def fork(self, force: bool = False, **kwargs) -> Self:
        EntityRegistry._logger.debug(f"Fork called on entity {self.id} (force={force}, modifications={kwargs})")
        
        # Retrieve cold snapshot to compare modifications.
        cold_snapshot = EntityRegistry.get_cold_snapshot(self.id)
        old_id = self.id
        if cold_snapshot is None:
            EntityRegistry._logger.debug(f"No cold snapshot found - registering new entity {self.id}")
            EntityRegistry.register(self)
            return self

        # Apply modifications and create new version if needed
        if not self._apply_modifications_and_create_version(cold_snapshot, force, **kwargs):
            return self
        
        # Register the new version.
        EntityRegistry.register(self)
        
        EntityRegistry._logger.debug(f"Fork complete: created new version {self.id} from {old_id}")
        return self

    @model_validator(mode='after')
    def register_entity(self) -> Self:
        """Register entity after creation/modification."""
        # First creation
        if not EntityRegistry.has_entity(self.id):
            EntityRegistry.register(self)
            return self
            
        # Check for modifications
        cold_snapshot = EntityRegistry.get_cold_snapshot(self.id)
        if cold_snapshot is not None and self.has_modifications(cold_snapshot):
            # Let fork handle the versioning
            self.fork()
            
        return self

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

    @classmethod
    def compare_entities(cls, entity1: 'Entity', entity2: Dict[str, Any]) -> bool:
        """Compare an entity with a snapshot dictionary."""
        # Get current entity's data excluding version fields
        data1 = entity1.entity_dump(exclude={'id', 'created_at', 'lineage_id', 'parent_id'})
        
        # Compare field by field
        all_keys = set(data1.keys()) | set(entity2.keys())
        for key in all_keys:
            val1 = data1.get(key)
            val2 = entity2.get(key)
            
            if val1 != val2:
                return False
                
        return True

########################################
# 3) Snapshot-based EntityRegistry
########################################

EType = TypeVar('EType', bound=Entity)

class EntityRegistry(BaseRegistry[EType]):
    """
    Registry for managing immutable snapshots of entities with lineage-based versioning.
    
    The registry stores cold snapshots of entities, while warm entities are modified in place.
    When an entity is registered:
    1. If it's a new entity, create and store a cold snapshot
    2. If it's an existing entity with modifications, use fork() to create a new version
    """
    _registry: Dict[UUID, EType] = {}
    _timestamps: Dict[UUID, datetime] = {}
    _inference_orchestrator: Optional[object] = None
    _lineages: Dict[UUID, List[UUID]] = {}
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.WARNING)
    _tracing_enabled: bool = True  # New state flag for controlling tracing

    @classmethod
    def has_entity(cls, entity_id: UUID) -> bool:
        """Check if an entity exists in the registry."""
        return entity_id in cls._registry

    @classmethod
    def get_cold_snapshot(cls, entity_id: UUID) -> Optional[EType]:
        """Get the cold snapshot of an entity by ID."""
        return cls._registry.get(entity_id)

    @classmethod
    def _store_cold_snapshot(cls, entity: EType) -> None:
        """Store a cold snapshot of an entity and update lineage tracking."""
        cold_snapshot = deepcopy(entity)
        cls._registry[entity.id] = cold_snapshot
        cls._timestamps[entity.id] = datetime.utcnow()
        
        # Initialize lineage tracking
        if hasattr(entity, 'lineage_id'):
            lineage_id = entity.lineage_id
            if lineage_id not in cls._lineages:
                cls._lineages[lineage_id] = []
            if entity.id not in cls._lineages[lineage_id]:
                cls._lineages[lineage_id].append(entity.id)
                cls._logger.debug(f"Added {entity.id} to lineage {lineage_id}")

    @classmethod
    def register(cls, entity: Union[EType, UUID]) -> Optional[EType]:
        """Register an entity or retrieve by UUID."""
        cls._logger.debug(f"Registering entity: {entity}")
        
        if isinstance(entity, UUID):
            cls._logger.debug(f"Entity is UUID, retrieving from registry: {entity}")
            return cls.get(entity)
            
        if not isinstance(entity, BaseModel):
            msg = f"Entity must be a Pydantic model instance, got {type(entity)}"
            cls._logger.error(msg)
            raise ValueError(msg)
            
        ent_id = entity.id
        cls._logger.debug(f"Processing entity with ID: {ent_id}")

        # First registration - create cold snapshot
        if ent_id not in cls._registry:
            cls._logger.debug(f"First registration for entity {ent_id}, creating cold snapshot")
            cls._store_cold_snapshot(entity)
            cls._logger.debug(f"Cold snapshot created for entity {ent_id}")
            return entity

        # Get existing cold snapshot
        cold_snapshot = cls._registry[ent_id]
        cls._logger.debug(f"Retrieved existing cold snapshot for entity {ent_id}")
        
        # Check for modifications and fork if needed
        if entity.has_modifications(cold_snapshot):
            cls._logger.debug(f"Entity {ent_id} has modifications, creating new version via fork")
            entity.fork()
            cls._logger.debug(f"Created new version for entity {ent_id}")
            return entity
            
        cls._logger.debug(f"No modifications detected for entity {ent_id}")
        return entity

    @classmethod
    def get(cls, entity_id: UUID, expected_type: Optional[Type[EType]] = None) -> Optional[EType]:
        """Retrieve a cold snapshot by its ID and return a copy."""
        cls._logger.debug(f"Retrieving entity {entity_id}")
        entity = cls._registry.get(entity_id)
        if entity is None:
            cls._logger.debug(f"Entity {entity_id} not found")
            return None
        if expected_type and not isinstance(entity, expected_type):
            cls._logger.error(f"Type mismatch for {entity_id}. Expected {expected_type.__name__}, got {type(entity).__name__}")
            return None
        # Return a copy of the cold snapshot with a new live_id
        warm_copy = deepcopy(entity)
        warm_copy.live_id = uuid4()  # New live instance gets new live_id
        return warm_copy

    @classmethod
    def list_by_type(cls, entity_type: Type[EType]) -> List[EType]:
        """List all cold snapshots of a given type."""
        cls._logger.debug(f"Listing entities of type {entity_type.__name__}")
        return [deepcopy(e) for e in cls._registry.values() if isinstance(e, entity_type)]

    @classmethod
    def get_many(cls, entity_ids: List[UUID], expected_type: Optional[Type[EType]] = None) -> List[EType]:
        """Retrieve multiple cold snapshots by their IDs."""
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
    def build_lineage_tree(cls, lineage_id: UUID) -> Dict[UUID, Dict[str, Any]]:
        """Builds a tree structure for a given lineage using cold snapshots."""
        version_ids = cls._lineages.get(lineage_id, [])
        if not version_ids:
            cls._logger.info(f"No versions found for lineage {lineage_id}")
            return {}
            
        # Find root node (node without parent)
        root_id = None
        for vid in version_ids:
            entity = cls._registry.get(vid)
            if not isinstance(entity, Entity):
                continue
            if not entity.parent_id:
                root_id = vid
                break
                
        if not root_id:
            cls._logger.error(f"No root node found for lineage {lineage_id}")
            return {}
            
        tree: Dict[UUID, Dict[str, Any]] = {}
        
        def build_subtree(node_id: UUID, depth: int = 0) -> None:
            """Recursively build the tree starting from a node."""
            entity = cls._registry.get(node_id)
            if not isinstance(entity, Entity):
                cls._logger.warning(f"Invalid entity found in lineage: {node_id}")
                return
                
            cls._logger.debug(f"Building subtree for node {node_id} at depth {depth}")
            
            # Calculate diff from parent if not root
            diff_from_parent = None
            if entity.parent_id:
                parent = cls._registry.get(entity.parent_id)
                if parent and isinstance(parent, Entity):
                    diff_from_parent = entity.compute_diff(parent).field_diffs
                    cls._logger.debug(f"Computed diff from parent {entity.parent_id}")
            
            # Add node to tree
            tree[node_id] = {
                "entity": entity,
                "children": [],
                "depth": depth,
                "parent_id": entity.parent_id,
                "created_at": entity.created_at,
                "data": entity.entity_dump(exclude={'id', 'created_at', 'lineage_id', 'parent_id'}),
                "diff_from_parent": diff_from_parent
            }
            
            # Add this node as child to parent
            if entity.parent_id and entity.parent_id in tree:
                tree[entity.parent_id]["children"].append(node_id)
                cls._logger.debug(f"Added {node_id} as child of {entity.parent_id}")
            
            # Find and process children
            for vid in version_ids:
                child = cls._registry.get(vid)
                if not isinstance(child, Entity):
                    continue
                if child.parent_id == node_id:
                    cls._logger.debug(f"Processing child {vid} of node {node_id}")
                    build_subtree(vid, depth + 1)
        
        # Build tree starting from root
        cls._logger.info(f"Starting tree build from root {root_id}")
        build_subtree(root_id)
        cls._logger.info(f"Completed tree build for lineage {lineage_id}")
        return tree

    @classmethod
    def get_lineage_tree_sorted(cls, lineage_id: UUID) -> Dict[str, Any]:
        """
        Returns a structured representation of the lineage tree, sorted by creation time.
        The returned structure contains:
        - nodes: Dict[UUID, Dict] - All nodes in the tree with their metadata and diffs
        - edges: List[Tuple[UUID, UUID]] - Parent->Child relationships
        - root: UUID - The root node ID
        - sorted_ids: List[UUID] - Node IDs sorted by creation time
        - diffs: Dict[UUID, Dict] - Changes from parent for each node (excluding root)
        """
        # First get all nodes in this lineage
        version_ids = cls._lineages.get(lineage_id, [])
        if not version_ids:
            cls._logger.warning(f"No versions found for lineage {lineage_id}")
            return {
                "nodes": {},
                "edges": [],
                "root": None,
                "sorted_ids": [],
                "diffs": {}
            }

        # Log current state for debugging
        cls._logger.debug(f"Building tree for lineage {lineage_id}")
        cls._logger.debug(f"Version IDs in lineage: {version_ids}")
        
        # Build the tree
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
        sorted_items = sorted(tree.items(), key=lambda kv: kv[1]["created_at"])
        sorted_ids = [vid for vid, _ in sorted_items]
        
        # Build edges list and collect diffs
        edges = []
        diffs = {}
        for vid, node in tree.items():
            if node["parent_id"]:
                edges.append((node["parent_id"], vid))
                if node["diff_from_parent"]:
                    diffs[vid] = node["diff_from_parent"]
                cls._logger.debug(f"Added edge: {node['parent_id']} -> {vid}")
        
        # Find root
        roots = [vid for vid, node in tree.items() if not node["parent_id"]]
        root = roots[0] if roots else None
        
        # Log final tree state for debugging
        cls._logger.debug(f"Final tree state:")
        cls._logger.debug(f"Nodes: {list(tree.keys())}")
        cls._logger.debug(f"Edges: {edges}")
        cls._logger.debug(f"Root: {root}")
        cls._logger.debug(f"Diffs: {diffs}")
        
        return {
            "nodes": tree,
            "edges": edges,
            "root": root,
            "sorted_ids": sorted_ids,
            "diffs": diffs
        }

    @classmethod
    def get_lineage_mermaid(cls, lineage_id: UUID) -> str:
        """Generate a Mermaid graph visualization of the lineage tree."""
        cls._logger.debug(f"Generating Mermaid diagram for lineage {lineage_id}")
        tree = cls.get_lineage_tree_sorted(lineage_id)
        if not tree or not tree["nodes"]:
            cls._logger.warning(f"No data available for lineage {lineage_id}")
            return "```mermaid\ngraph TD\n  No data available\n```"
            
        mermaid_lines = ["```mermaid", "graph TD"]
        
        def format_value(value: Any) -> str:
            """Format a value for display in Mermaid."""
            if isinstance(value, (list, tuple)):
                return f"[{len(value)}]"
            elif isinstance(value, dict):
                return f"{{{len(value)}}}"
            else:
                val_str = str(value)[:20]
                return val_str + "..." if len(str(value)) > 20 else val_str
        
        def add_node(node_id: UUID, node_data: Dict[str, Any]) -> None:
            """Add a node to the Mermaid diagram with proper formatting."""
            entity = node_data["entity"]
            node_type = type(entity).__name__
            short_id = str(node_id)[:8]
            
            # For root node, show initial state
            if not node_data["parent_id"]:
                data = node_data["data"]
                data_summary = [
                    f"{key}={format_value(value)}"
                    for key, value in data.items()
                    if not isinstance(value, (bytes, bytearray))
                ][:3]  # Limit to 3 fields
                if len(data) > 3:
                    data_summary.append(f"...({len(data)-3} more)")
                data_text = "\\n" + ", ".join(data_summary) if data_summary else ""
                mermaid_lines.append(f"  {node_id}[\"{node_type}\\n{short_id}{data_text}\"]")
            else:
                # For non-root nodes, show type, ID, and modification count
                diff = node_data.get("diff_from_parent", {})
                mod_count = len(diff) if diff else 0
                mermaid_lines.append(f"  {node_id}[\"{node_type}\\n{short_id}\\n({mod_count} changes)\"]")
        
        def add_edge(parent_id: UUID, child_id: UUID, node_data: Dict[str, Any]) -> None:
            """Add an edge to the Mermaid diagram with diff information."""
            diff = node_data.get("diff_from_parent", {})
            if not diff:
                mermaid_lines.append(f"  {parent_id} --> {child_id}")
                return
                
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
                mermaid_lines.append(f"  {parent_id} -->|\"{diff_text}\"| {child_id}")
            else:
                mermaid_lines.append(f"  {parent_id} --> {child_id}")
        
        # Add all nodes first
        for node_id, node_data in tree["nodes"].items():
            add_node(node_id, node_data)
        
        # Then add all edges
        for edge in tree["edges"]:
            parent_id, child_id = edge
            node_data = tree["nodes"][child_id]
            add_edge(parent_id, child_id, node_data)
        
        mermaid_lines.append("```")
        return "\n".join(mermaid_lines)

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
        cls._timestamps.clear()
        cls._lineages.clear()

    @classmethod
    def set_tracing_enabled(cls, enabled: bool) -> None:
        """Enable or disable decorator-based entity tracing."""
        cls._tracing_enabled = enabled
        cls._logger.debug(f"Entity tracing {'enabled' if enabled else 'disabled'}")

    @classmethod
    def is_tracing_enabled(cls) -> bool:
        """Check if decorator-based entity tracing is enabled."""
        return cls._tracing_enabled

########################################
# 4) Decorators
########################################

# Type variables for decorators
T_dec = TypeVar("T_dec")
PS = ParamSpec("PS")
RT = TypeVar("RT")

def _collect_entities(args: tuple, kwargs: dict) -> Dict[int, Entity]:
    """Helper to collect all Entity instances from args and kwargs with their memory ids."""
    entities = {}
    
    # Scan args
    for arg in args:
        if isinstance(arg, Entity):
            entities[id(arg)] = arg
        elif isinstance(arg, (list, tuple, set)):
            for item in arg:
                if isinstance(item, Entity):
                    entities[id(item)] = item
        elif isinstance(arg, dict):
            for item in arg.values():
                if isinstance(item, Entity):
                    entities[id(item)] = item
    
    # Scan kwargs
    for arg in kwargs.values():
        if isinstance(arg, Entity):
            entities[id(arg)] = arg
        elif isinstance(arg, (list, tuple, set)):
            for item in arg:
                if isinstance(item, Entity):
                    entities[id(item)] = item
        elif isinstance(arg, dict):
            for item in arg.values():
                if isinstance(item, Entity):
                    entities[id(item)] = item
    
    return entities

def _check_and_fork_modified(entity: Entity) -> None:
    """Helper to check if an entity is modified and fork it if needed."""
    cold_snapshot = EntityRegistry.get_cold_snapshot(entity.id)
    if cold_snapshot is not None and entity.has_modifications(cold_snapshot):
        EntityRegistry._logger.debug(f"Entity {entity.id} has modifications, creating new version")
        entity.fork()

def entity_tracer(func):
    """
    Decorator to trace entity modifications and handle versioning.
    Automatically detects and handles all Entity instances in arguments.
    Works with both sync and async functions.
    
    If tracing is disabled in EntityRegistry, this decorator becomes a pass-through.
    """
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Skip tracing if disabled
            if not EntityRegistry.is_tracing_enabled():
                return await func(*args, **kwargs)

            # Original tracing logic
            entities = _collect_entities(args, kwargs)
            for entity in entities.values():
                _check_and_fork_modified(entity)
            
            result = await func(*args, **kwargs)
            
            for entity in entities.values():
                _check_and_fork_modified(entity)
                
            if isinstance(result, Entity) and id(result) in entities:
                return entities[id(result)]
            return result
            
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Skip tracing if disabled
            if not EntityRegistry.is_tracing_enabled():
                return func(*args, **kwargs)

            # Original tracing logic
            entities = _collect_entities(args, kwargs)
            for entity in entities.values():
                _check_and_fork_modified(entity)
            
            result = func(*args, **kwargs)
            
            for entity in entities.values():
                _check_and_fork_modified(entity)
                
            if isinstance(result, Entity) and id(result) in entities:
                return entities[id(result)]
            return result
            
        return sync_wrapper
