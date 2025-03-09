# MarketInference Project

## Useful Commands
- **Build/Install**: `pip install -e .`
- **Run Tests**: `pytest tests/threads_tests --asyncio-mode=auto`
- **Run Single Test File**: `pytest tests/threads_tests/test_file_name.py -v --asyncio-mode=auto`
- **Typecheck**: TBD
- **Lint**: TBD

## Project Structure
- `minference/`: Main package 
  - `clients/`: API client interfaces
  - `ecs/`: Entity Component System
  - `threads/`: Threading implementation
- `examples/`: Usage examples

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

## SQL Implementation Issues

The current SQL backend implementation uses SQLModel, which causes several problems:

### Current Issues with SQLModel
1. **Code Generation Compatibility**: SQLModel has a confusing syntax that mixes features from both SQLAlchemy and Pydantic, making it difficult for LLMs to generate correct code
2. **API Complexity**: Inconsistent API that requires special knowledge about how SQLModel's internals work
3. **Relationship Management**: Overly complex relationship handling that doesn't leverage SQLAlchemy's strengths
4. **Versioning Complexity**: The implementation doesn't cleanly separate SQL storage concerns from domain entity concerns

### Refactoring Approach
The planned refactoring will:

1. **Use Pure SQLAlchemy**: Replace SQLModel with standard SQLAlchemy ORM for better code clarity and maintainability
2. **Simplify Relationship Handling**: Use SQLAlchemy's relationship patterns more directly
3. **Improve Type Safety**: Maintain strong typing while simplifying conversion between entities and database models
4. **Enhance Versioning**: Better versioning support with cleaner relationships between entity versions

### Key Implementation Changes
- Move from table=True declarations to explicit SQLAlchemy models
- Replace Field with Column for clearer database column definitions
- Implement cleaner to_entity/from_entity conversion methods
- Better handle UUID conversion between Python UUIDs and database string representations
- Use proper association tables for many-to-many relationships
- Add polymorphic entity support for tool implementations

## Current Development: Systematic Testing

We are implementing a systematic testing approach to ensure a smooth migration from SQLModel to pure SQLAlchemy:

### Testing Strategy
1. **Start with in-memory tests**: Create a comprehensive test suite for the in-memory implementation first
2. **Define relationship patterns**: Systematically test all relationship types (one-to-one, one-to-many, many-to-many, hierarchical)
3. **Develop SQLAlchemy backend**: Once in-memory tests pass, implement and test the SQLAlchemy backend against the same test suite
4. **Apply to production models**: Refactor the production models based on proven patterns from the test suite

### Test Location
Tests are being developed in the `/tests/ecs_tests/` directory with:
- `DESIGN.md`: Detailed test planning document
- `conftest.py`: Test fixtures and mock entity classes 
- `test_basic_operations.py`: Tests for basic entity functionality
- `test_relationships.py`: Tests for entity relationships and change propagation

### Type Safety Approach

The test suite has been designed with strong type safety in mind:

1. **Explicit Type Annotations**: All functions, fixtures, and variables have explicit type annotations
2. **Null Safety**: Comprehensive null checks before accessing attributes from potentially null references
3. **Type Casting**: Strategic use of `cast()` for precise type control when generic Entity types are returned
4. **Forward References**: Careful handling of circular references in entity relationships
5. **Default Values**: Using `Field(default_factory=list)` rather than mutable default arguments

#### Type Safety Best Practices

When working with entities, always follow these guidelines to avoid type safety issues:

1. **Custom get() methods**: Override the `get()` classmethod in every entity subclass to return the correct type:
   ```python
   @classmethod
   def get(cls, entity_id: UUID) -> Optional[T_YourEntity]:
       entity = super().get(entity_id)
       return cast(Optional[T_YourEntity], entity)
   ```

2. **Type Variables**: Define type variables for entity subclasses:
   ```python
   T_YourEntity = TypeVar('T_YourEntity', bound='YourEntity')
   ```

3. **Cast Registry Results**: Always cast results from the EntityRegistry methods:
   ```python
   result = EntityRegistry.register(entity)
   if result:
       entity = cast(YourEntityType, result)
   ```

4. **Null Checks**: Always check for None before using EntityRegistry results:
   ```python
   entity_result = EntityRegistry.register(entity)
   if not entity_result:
       return  # Handle error
   entity = cast(YourEntityType, entity_result)
   ```

5. **Collection Properties**: For properties that return collections, use explicit typing and null checks:
   ```python
   @property
   def items(self) -> List[Item]:
       result: List[Item] = []
       for item_id in self.item_ids:
           if item_id:
               item = Item.get(item_id)
               if item:
                   result.append(item)
       return result
   ```

#### Dependency Graph Best Practices

When working with the EntityDependencyGraph class, follow these guidelines:

1. **Avoid Parameter Redeclaration**: Use different parameter names in method signatures and local variables:
   ```python
   # AVOID:
   def build_graph(self, root_entity, is_entity_func=None):
       # Redeclaration of parameter name
       if is_entity_func is None:
           def is_entity_func(obj):  # Error: parameter redeclaration
               return hasattr(obj, "ecs_id")
   
   # GOOD:
   def build_graph(self, root_entity, is_entity_check=None):
       # Different name for local function
       if is_entity_check is None:
           def is_entity_func(obj):  # OK: different name
               return hasattr(obj, "ecs_id")
       else:
           is_entity_func = is_entity_check
   ```

2. **Use Optional for Nullable Lists**: For parameters that can be None or a list, use Optional[List[T]]:
   ```python
   # AVOID:
   def add_entity(self, entity: Any, dependencies: List[Any] = None): 
       # Error: None is not compatible with List[Any]
   
   # GOOD:
   def add_entity(self, entity: Any, dependencies: Optional[List[Any]] = None):
       # OK: Optional makes it clear the parameter can be None
   ```

3. **Avoid Method Redeclaration**: Don't define the same method twice in a class:
   ```python
   # AVOID:
   def get_dependent_ids(self, entity_id):
       # First implementation...
   
   def get_dependent_ids(self, entity_id):  # Error: method redeclaration
       # Second implementation...
   
   # GOOD:
   def get_dependent_ids(self, entity_id):
       # Single implementation
   ```

This matches the type safety focus of the main codebase and ensures the tests can catch type-related issues early. The tests systematically verify key ECS functionality:

- Entity creation and registration
- Modification detection
- Entity forking and versioning
- Hierarchical relationships and change propagation
- Cold snapshots vs. warm copies

This approach ensures that we validate the core ECS functionality across different storage backends before tackling the more complex production models.

## Completed Work

### Comprehensive Thread Tests
We've implemented comprehensive tests for all entity types in the threads module:

1. **Core Entity Tests**:
   - `ChatThread` - Main conversation manager
   - `ChatMessage` - Individual messages with different roles
   - `CallableTool` - Function-based tool execution
   - `StructuredTool` - Schema-based tool validation
   - `LLMConfig` - LLM client configuration
   - `SystemPrompt` - Reusable system prompts
   - `Usage` - Token usage tracking
   - `GeneratedJsonObject` - Structured JSON data
   - `RawOutput` & `ProcessedOutput` - LLM response handling

2. **Integration Tests**:
   - Workflow management
   - Tool execution
   - Message threading
   - Sequential tool pipelines

### Best Practices Implemented

1. **Modern Timezone-Aware Datetimes**:
   - Replaced all uses of deprecated `datetime.utcnow()` with `datetime.now(timezone.utc)`
   - Ensures consistent timezone handling across all timestamps

2. **Updated Pydantic Serialization**:
   - Migrated from deprecated Pydantic V2 `json_encoders` to modern serialization approach
   - Using `model_config` with `ser_json_bytes` and `ser_json_timedelta` instead
   - Better compatibility with latest Pydantic versions

3. **Improved Dependency Handling**:
   - Enhanced entity dependency graph for better nested entity tracking
   - Proper cycle detection in entity references
   - Topological sorting for safe bottom-up processing

4. **Async/Await Support**:
   - Proper async function handling in entity tracing
   - Careful management of async context for tool execution
   - Support for both sync and async workflows

### Code Quality Standards

All code follows these standards:
1. **Type Safety**: Complete type annotations with proper generics
2. **Null Safety**: Consistent null checking before attribute access
3. **Error Handling**: Proper exception propagation and logging
4. **Testability**: Clean separation of concerns for easier testing
5. **Documentation**: Clear docstrings and comments

## Future Improvements

### 1. Enhanced Entity Tracer
The `entity_tracer` decorator should be improved to better handle newly created entities:

```python
# Current behavior only checks if result is an entity that was in arguments:
if isinstance(result, Entity) and id(result) in entities:
    return entities[id(result)]

# Should also detect and register new entities that were created inside the function:
if isinstance(result, Entity) and id(result) not in entities:
    # Register the new entity if not already registered
    # Consider auto-registering hierarchies of new entities
```

This would ensure that new entities created within traced functions are properly tracked and versioned without requiring explicit registration.

### 2. Trace Severity Levels
Add configurable severity levels to the entity tracing system:

```python
# Define trace severity enum
class TraceSeverity(Enum):
    NONE = 0     # No tracing
    CRITICAL = 1 # Only trace critical operations (e.g., root entity modifications)
    NORMAL = 2   # Trace significant operations (default)
    VERBOSE = 3  # Trace all operations including nested entities

# Enhanced decorator with severity parameter
def entity_tracer(func=None, *, severity: TraceSeverity = TraceSeverity.NORMAL):
    # Implementation details...
```

This would allow global control over tracing granularity through:
```python
EntityRegistry.set_trace_severity(TraceSeverity.VERBOSE)  # More detailed tracing
EntityRegistry.set_trace_severity(TraceSeverity.CRITICAL) # Minimal tracing for production
```

### 3. Logging System Overhaul
Implement a comprehensive logging system following best practices:

1. **Centralized Logger Configuration**:
   ```python
   class LogManager:
       _instance = None
       _handlers = {}
       _log_level = logging.INFO
       
       @classmethod
       def configure(cls, level=logging.INFO, log_file=None, max_size=10*1024*1024):
           cls._log_level = level
           # Configure handlers, formatters, etc.
   ```

2. **Memory-Efficient Logging**:
   - Implement log rotation with configurable retention
   - Add methods to clear logs to save memory
   - Support streaming to external systems

3. **Context-Aware Logging**:
   - Add entity ID and operation context to log records
   - Support structured logging for better searchability
   - Add correlation IDs for tracing operations across the system

4. **Integration with Entity System**:
   - Log important entity lifecycle events
   - Track entity lineage through logs
   - Support debugging complex relationships

This would provide better visibility into system behavior while maintaining good performance.

## Notes
- Current branch: iri_claude_code_sql_refactor
- Main branch: main