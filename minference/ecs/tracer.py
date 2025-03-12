from typing import Any, Dict, List, Set, Callable
import logging
from minference.ecs.entity import Entity, EntityRegistry
from functools import wraps

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