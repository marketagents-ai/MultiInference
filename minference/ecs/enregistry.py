import logging
from typing import Any, Dict, List, Optional, Type, Set, Union, Callable
from uuid import UUID
from functools import wraps

from minference.ecs.entity import Entity, BaseRegistry
from minference.ecs.storage import EntityStorage, InMemoryEntityStorage

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
    def merge_entity(cls, entity: Entity) -> Optional[Entity]:
        """
        Special method for SQL storage to merge an entity that might be attached to another session.
        If using in-memory storage, this just calls register.
        
        Args:
            entity: Entity to merge
            
        Returns:
            The merged entity if successful, None otherwise
        """
        # Check if we're using SQL storage
        storage_info = cls.get_registry_status()
        if storage_info.get('storage') == 'sql' and hasattr(cls._storage, 'merge_entity'):
            # Use hasattr to check for merge_entity method at runtime
            merge_method = getattr(cls._storage, 'merge_entity')
            return merge_method(entity)
        # Otherwise, use normal registration
        return cls.register(entity)
        
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

def _check_and_process_entities(entities: Dict[int, Entity], fork_if_modified: bool = True) -> None:
    """
    Check entities for modifications and optionally fork them.
    Process in bottom-up order (nested entities first).
    """
    logger = logging.getLogger("EntityProcessing")
    logger.info(f"Processing {len(entities)} entities, fork_if_modified={fork_if_modified}")
    
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
                logger.debug(f"Dependency: {type(entity).__name__}({entity.ecs_id}) depends on {type(sub).__name__}({sub.ecs_id})")
    
    logger.debug(f"Built dependency graph with {len(dependency_graph)} nodes")
    
    # Topological sort (process leaves first)
    processed: Set[int] = set()
    
    def process_entity(entity_id: int) -> None:
        if entity_id in processed:
            logger.debug(f"Entity {entity_id} already processed, skipping")
            return
            
        # Process dependencies first
        for dep_id in dependency_graph[entity_id]:
            if dep_id not in processed:
                logger.debug(f"Processing dependency {dep_id} first")
                process_entity(dep_id)
                
        # Process this entity
        entity = entities[entity_id]
        logger.info(f"Processing entity {type(entity).__name__}({entity.ecs_id})")
        cold = EntityRegistry.get_cold_snapshot(entity.ecs_id)
        
        if cold:
            needs_fork, modified_entities = entity.has_modifications(cold)
            if needs_fork and fork_if_modified:
                logger.info(f"Entity {type(entity).__name__}({entity.ecs_id}) has modifications, forking")
                forked = entity.fork()
                logger.debug(f"Forked to new entity {forked.ecs_id}")
            else:
                logger.debug(f"Entity {type(entity).__name__}({entity.ecs_id}) has no modifications or fork_if_modified=False")
        else:
            logger.debug(f"No cold snapshot found for entity {entity.ecs_id}")
            
        processed.add(entity_id)
        logger.debug(f"Marked entity {entity_id} as processed")
    
    # Process all entities
    for entity_id in entities:
        if entity_id not in processed:
            logger.debug(f"Starting processing for entity {entity_id}")
            process_entity(entity_id)
    
    logger.info(f"Finished processing {len(processed)}/{len(entities)} entities")


def entity_tracer(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to trace entity modifications and handle versioning.
    Automatically detects and handles all Entity instances in arguments.
    Works with both sync and async functions, and both storage types.
    """
    logger = logging.getLogger("EntityTracer")
    logger.info(f"Decorating function {func.__name__} with entity_tracer")
    
    # Handle detection of async functions safely
    is_async = False
    try:
        # Try to import inspect locally to avoid any module conflicts
        import inspect as local_inspect
        is_async = local_inspect.iscoroutinefunction(func)
    except (ImportError, AttributeError):
        # Fallback method if inspect.iscoroutinefunction is not available
        is_async = hasattr(func, '__await__') or (hasattr(func, '__code__') and func.__code__.co_flags & 0x80)
    
    logger.debug(f"Function {func.__name__} is {'async' if is_async else 'sync'}")
    
    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info(f"Entering async wrapper for {func.__name__}")
        
        # Collect all entities from inputs
        entities = _collect_entities(args, kwargs)
        logger.info(f"Collected {len(entities)} entities from arguments")
        
        # Get storage type to adjust behavior
        from __main__ import EntityRegistry
        storage_info = EntityRegistry.get_registry_status()
        using_sql_storage = storage_info.get('storage') == 'sql'
        logger.debug(f"Storage type: {'SQL' if using_sql_storage else 'In-Memory'}")
        
        # Check for modifications before call
        fork_count = 0
        for entity_id, entity in entities.items():
            logger.debug(f"Checking entity {type(entity).__name__}({entity.ecs_id}) before function call")
            cold_snapshot = EntityRegistry.get_cold_snapshot(entity.ecs_id)
            if cold_snapshot:
                # Special handling based on storage type
                if using_sql_storage:
                    needs_fork, modified = entity.has_modifications(cold_snapshot)
                    if needs_fork:
                        logger.info(f"Entity {type(entity).__name__}({entity.ecs_id}) needs fork - forking before call")
                        entity.fork()
                        fork_count += 1
                else:
                    # Simpler check for in-memory mode for better tracing
                    needs_fork, _ = entity.has_modifications(cold_snapshot)
                    if needs_fork:
                        logger.info(f"Entity {type(entity).__name__}({entity.ecs_id}) needs fork - forking before call (in-memory mode)")
                        entity.fork()
                        fork_count += 1
            else:
                logger.debug(f"No cold snapshot found for entity {entity.ecs_id}")
        
        logger.info(f"Forked {fork_count} entities before calling {func.__name__}")

        # Call the function
        logger.debug(f"Calling async function {func.__name__}")
        result = await func(*args, **kwargs)
        logger.debug(f"Function {func.__name__} returned: {type(result)}")

        # Check for modifications after call - same logic as before
        after_fork_count = 0
        for entity_id, entity in entities.items():
            logger.debug(f"Checking entity {type(entity).__name__}({entity.ecs_id}) after function call")
            cold_snapshot = EntityRegistry.get_cold_snapshot(entity.ecs_id)
            if cold_snapshot:
                if using_sql_storage:
                    needs_fork, modified = entity.has_modifications(cold_snapshot)
                    if needs_fork:
                        logger.info(f"Entity {type(entity).__name__}({entity.ecs_id}) needs fork - forking after call")
                        entity.fork()
                        after_fork_count += 1
                else:
                    needs_fork, _ = entity.has_modifications(cold_snapshot)
                    if needs_fork:
                        logger.info(f"Entity {type(entity).__name__}({entity.ecs_id}) needs fork - forking after call (in-memory mode)")
                        entity.fork()
                        after_fork_count += 1
            else:
                logger.debug(f"No cold snapshot found for entity {entity.ecs_id} after call")
            
        logger.info(f"Forked {after_fork_count} entities after calling {func.__name__}")
        
        # If result is an entity, handle it appropriately
        if isinstance(result, Entity):
            # If it was an input entity that was modified, return the forked version
            if id(result) in entities:
                logger.info(f"Result is an entity that was in arguments, returning most recent version")
                return entities[id(result)]
            # If it's a newly created entity, register it automatically
            elif not EntityRegistry.has_entity(result.ecs_id):
                logger.info(f"Result is a newly created entity, registering it automatically")
                return EntityRegistry.register(result)

        logger.debug(f"Exiting async wrapper for {func.__name__}")
        return result
    
    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info(f"Entering sync wrapper for {func.__name__}")
        
        # Collect all entities from inputs
        entities = _collect_entities(args, kwargs)
        logger.info(f"Collected {len(entities)} entities from arguments")
        
        # Get storage type to adjust behavior
        from __main__ import EntityRegistry
        storage_info = EntityRegistry.get_registry_status()
        using_sql_storage = storage_info.get('storage') == 'sql'
        logger.debug(f"Storage type: {'SQL' if using_sql_storage else 'In-Memory'}")
        
        # Check for modifications before call
        fork_count = 0
        for entity_id, entity in entities.items():
            logger.debug(f"Checking entity {type(entity).__name__}({entity.ecs_id}) before function call")
            cold_snapshot = EntityRegistry.get_cold_snapshot(entity.ecs_id)
            if cold_snapshot:
                # Special handling based on storage type
                if using_sql_storage:
                    needs_fork, modified = entity.has_modifications(cold_snapshot)
                    if needs_fork:
                        logger.info(f"Entity {type(entity).__name__}({entity.ecs_id}) needs fork - forking before call")
                        entity.fork()
                        fork_count += 1
                else:
                    # Simpler check for in-memory mode for better tracing
                    needs_fork, _ = entity.has_modifications(cold_snapshot)
                    if needs_fork:
                        logger.info(f"Entity {type(entity).__name__}({entity.ecs_id}) needs fork - forking before call (in-memory mode)")
                        entity.fork()
                        fork_count += 1
            else:
                logger.debug(f"No cold snapshot found for entity {entity.ecs_id}")
        
        logger.info(f"Forked {fork_count} entities before calling {func.__name__}")

        # Call the function
        logger.debug(f"Calling sync function {func.__name__}")
        result = func(*args, **kwargs)
        logger.debug(f"Function {func.__name__} returned: {type(result)}")

        # Check for modifications after call - same logic as before
        after_fork_count = 0
        for entity_id, entity in entities.items():
            logger.debug(f"Checking entity {type(entity).__name__}({entity.ecs_id}) after function call")
            cold_snapshot = EntityRegistry.get_cold_snapshot(entity.ecs_id)
            if cold_snapshot:
                if using_sql_storage:
                    needs_fork, modified = entity.has_modifications(cold_snapshot)
                    if needs_fork:
                        logger.info(f"Entity {type(entity).__name__}({entity.ecs_id}) needs fork - forking after call")
                        entity.fork()
                        after_fork_count += 1
                else:
                    needs_fork, _ = entity.has_modifications(cold_snapshot)
                    if needs_fork:
                        logger.info(f"Entity {type(entity).__name__}({entity.ecs_id}) needs fork - forking after call (in-memory mode)")
                        entity.fork()
                        after_fork_count += 1
            else:
                logger.debug(f"No cold snapshot found for entity {entity.ecs_id} after call")
            
        logger.info(f"Forked {after_fork_count} entities after calling {func.__name__}")
        
        # If result is an entity, handle it appropriately
        if isinstance(result, Entity):
            # If it was an input entity that was modified, return the forked version
            if id(result) in entities:
                logger.info(f"Result is an entity that was in arguments, returning most recent version")
                return entities[id(result)]
            # If it's a newly created entity, register it automatically
            elif not EntityRegistry.has_entity(result.ecs_id):
                logger.info(f"Result is a newly created entity, registering it automatically")
                return EntityRegistry.register(result)

        logger.debug(f"Exiting sync wrapper for {func.__name__}")
        return result
    
    # Use the appropriate wrapper based on whether the function is async
    if is_async:
        return async_wrapper
    else:
        return sync_wrapper