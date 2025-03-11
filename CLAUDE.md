# MarketInference Project

## Useful Commands
- **Build/Install**: `pip install -e .`
- **Run Tests**: `pytest tests/threads_tests --asyncio-mode=auto`
- **Run Single Test File**: `pytest tests/threads_tests/test_file_name.py -v --asyncio-mode=auto`
- **Typecheck**: TBD
- **Lint**: TBD
- **Virtual Environment**: `/Users/tommasofurlanello/Documents/Dev/MarketInference/.venv`

## Project Structure
- `minference/`: Main package 
  - `clients/`: API client interfaces
  - `ecs/`: Entity Component System
  - `threads/`: Threading implementation
- `examples/`: Usage examples
- `tests/`: Test suite
## ECS System (Entity Component System)
The ECS system provides hierarchical version control for entities with the following key features:

### Core Concepts
1. **Entity Identity and State**:
   - Entities have both `ecs_id` (version identifier) and `live_id` (memory state identifier)
   - Entities are hashable by (id, live_id) combination
   - Cold snapshots are stored versions, warm copies are in-memory working versions
   - Same Python object reference is maintained while tracking versions

2. **Hierarchical Structure**:
   - Entities can contain other entities (sub-entities) in fields, lists, or dictionaries
   - The `get_sub_entities()` method recursively discovers all nested entities
   - Changes in sub-entities trigger parent entity versioning

3. **Dependency Graph**:
   - The `EntityDependencyGraph` class explicitly tracks relationships between entities
   - Handles circular references gracefully through cycle detection
   - Provides topological sorting for bottom-up processing of entity trees
   - Enables more explicit control over entity relationships than recursive discovery

4. **Modification Detection**:
   - `has_modifications()` performs deep comparison of entity trees
   - Returns both whether changes exist and the set of modified entities
   - Handles nested changes automatically through recursive comparison

5. **Forking Process**:
   - When changes are detected, affected entities get new IDs
   - Parent entities automatically fork when sub-entities change
   - All changes happen in memory first, then are committed to storage
   - Uses topological sorting from dependency graph for bottom-up forking

6. **Storage Layer**:
   - In-memory storage: `InMemoryEntityStorage` uses Python's object references
   - SQL storage: `SqlEntityStorage` handles SQL database operations (optional)
   - `EntityRegistry` provides a facade for consistent access to storage

### Key Components
- **Entity**: Base class for all entities with versioning support
- **EntityRegistry**: Static facade for storage operations
- **EntityStorage**: Protocol defining storage operations
- **entity_tracer**: Decorator to automatically track entity modifications

### Usage Example
```python
# Create and modify an entity
entity = ComplexEntity(nested=SubEntity(...))
entity.nested.value = "new"

# Check for modifications
if entity.has_modifications(stored_version):
    new_version = entity.fork()  # Creates new IDs for changed entities
    
# Store the entity
EntityRegistry.register(new_version)  # Stores all sub-entities
```

## Thread System
The Thread system builds on top of the ECS to model chat interactions with LLMs:

### Core Components
1. **ChatThread**: Main entity that manages a conversation
   - Tracks message history
   - Handles LLM configuration and requests
   - Manages tool execution

2. **ChatMessage**: Entity representing a single message in a thread
   - Different roles (user, assistant, tool, system)
   - Can be linked to tools for execution
   - Stores usage statistics and other metadata

3. **Tool Entities**:
   - **CallableTool**: Represents executable functions with schemas
   - **StructuredTool**: Validates outputs against JSON schemas

4. **LLMConfig**: Configuration for language model interactions
   - Specifies client, model, and response format settings
   - Handles different response types (text, JSON, tools)

5. **InferenceOrchestrator**: Coordinates parallel LLM API calls
   - Manages rate limits across different providers
   - Processes outputs and executes tools

### Workflow
1. Create ChatThread with configuration and tools
2. Add user messages
3. Run inference using the orchestrator
4. Process LLM responses and execute tools 
5. All state changes automatically versioned through ECS

### Integration with ECS
- All entities inherit from the base Entity class
- Uses entity_tracer for automatic versioning
- Maintains history as versioned Entity objects
