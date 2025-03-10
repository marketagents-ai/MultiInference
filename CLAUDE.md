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

## Current Implementation Status

We have implemented a pure SQLAlchemy-based storage backend with the following components:

1. **EntityBase**: Abstract base class for all entity tables with common columns for versioning
2. **BaseEntitySQL**: Generic fallback table for storing entities without specific models
3. **SqlEntityStorage**: Main storage implementation that provides all operations required by the EntityStorage protocol

### Key Accomplishments

1. **Complete SQLAlchemy Implementation**: We've created a full implementation of the EntityStorage protocol using pure SQLAlchemy ORM with modern typing features.

2. **Entity-ORM Mapping System**: We've designed a flexible system that maps entity classes to their corresponding ORM models:
   - Explicit mapping provided at initialization for type safety
   - Inheritance-aware mapping that finds the most specific ORM model
   - Memory-efficient caching for repeated lookups

3. **Relationship Handling**: We've implemented support for all relationship types:
   - One-to-one relationships (via direct foreign keys)
   - One-to-many relationships (via collections)
   - Many-to-many relationships (via association tables)
   - Self-referential relationships
   
4. **Test Entity Models**: We've created test entities and corresponding ORM models that showcase the different relationship patterns:
   - SimpleEntity - Basic entity with primitive fields
   - NestedEntity - Entity containing another entity
   - ParentEntity/ChildEntity - Many-to-many relationship
   - ComplexEntity - Entity with multiple relationship types

5. **Comprehensive Testing**: We've developed tests that verify:
   - Roundtrip conversion (Entity→ORM→Entity)
   - Entity identity preservation
   - Relationship integrity
   - Hierarchical storage and retrieval

### Current Issues and Challenges

1. ✅ **UUID Serialization**: Fixed the issue with UUID serialization in JSON columns by:
   - Converting UUID objects to strings in all `from_entity` methods
   - Converting string representations back to UUID objects in `to_entity` methods
   - Affecting all entity classes that store data in JSON columns
   
   The issue was:
   ```
   TypeError: Object of type UUID is not JSON serializable
   ```
   
   The solution implemented:
   ```python
   # In from_entity methods:
   str_old_ids = [str(uid) for uid in entity.old_ids] if entity.old_ids else []
   
   # In to_entity methods:
   uuid_old_ids = []
   if self.old_ids:
       for old_id in self.old_ids:
           if isinstance(old_id, str):
               uuid_old_ids.append(UUID(old_id))
           elif isinstance(old_id, UUID):
               uuid_old_ids.append(old_id)
   ```

2. **Database vs Entity IDs**: We need to clearly separate database row IDs (auto-incrementing integers) from entity versioning UUIDs. The current implementation properly creates separate IDs but needs better handling during serialization.

3. **Table Name Management**: Table names need to be explicitly declared in each ORM model class to avoid SQLAlchemy conflicts.

4. **Relationship Complexity**: Managing different types of relationships (one-to-one, one-to-many, many-to-many) requires careful implementation of relationship handling methods on each ORM model.

### UUID Serialization Implementation

We implemented a solution for the UUID serialization issue that was causing test failures. We chose the Type Conversion approach in all entity model classes:

1. ✅ **Type Conversion in from_entity and to_entity Methods**:
   We applied this approach to all ORM model classes in the system. The implementation:
   
   ```python
   @classmethod
   def from_entity(cls, entity: Entity) -> 'EntityBaseSQL':
       # Convert UUID lists to string lists
       str_old_ids = [str(uid) for uid in entity.old_ids] if entity.old_ids else []
       
       return cls(
           ecs_id=entity.ecs_id,
           lineage_id=entity.lineage_id,
           parent_id=entity.parent_id,
           created_at=entity.created_at,
           old_ids=str_old_ids,  # Store as strings
           # ... other fields
       )
   
   def to_entity(self) -> Entity:
       # Convert string representation back to UUID objects
       uuid_old_ids = []
       if self.old_ids:
           for old_id in self.old_ids:
               if isinstance(old_id, str):
                   uuid_old_ids.append(UUID(old_id))
               elif isinstance(old_id, UUID):
                   uuid_old_ids.append(old_id)
       
       return EntityType(
           ecs_id=self.ecs_id,
           lineage_id=self.lineage_id,
           parent_id=self.parent_id,
           created_at=self.created_at,
           old_ids=uuid_old_ids,  # Use converted UUID objects
           # ... other fields
       )
   ```

This approach was applied to all ORM model classes in `minference/threads/sql_models.py` and the `BaseEntitySQL` class in `minference/ecs/entity.py`. 

The benefit of this approach is:
1. It's explicit and easy to understand
2. It keeps the serialization logic in the conversion methods
3. It doesn't require custom TypeDecorators
4. It's compatible with all database backends

All 32 tests in the SQL test suite now pass successfully, confirming the fix is working correctly.

### Next Steps

1. ✅ Implement the UUID serialization fix in the ORM models - **COMPLETED**
2. Improve type safety in the SqlEntityStorage class, particularly for the _get_orm_class method
3. Complete the test suite to cover all core operations:
   - Basic entity storage and retrieval ✓
   - Nested entity relationships ✓
   - Many-to-many relationships ✓
   - Entity modification and versioning ✓
   - Querying by type ✓
   - Batch operations ✓

4. Integrate the implementation with the main codebase by replacing the SQLModel-based storage
5. Evaluate performance with larger datasets and optimize as needed

### Test Status

All tests in the SQL module are now passing. This includes:
- Entity conversion tests
- Storage interface tests
- ORM table creation tests
- Entity identity preservation tests
- SQL registry integration tests
- Thread entity model tests

This represents a significant milestone in the migration from SQLModel to pure SQLAlchemy, with all core functionality correctly implemented and tested.

## Current Development: SQLAlchemy Migration

We are implementing a migration from SQLModel to pure SQLAlchemy ORM for the entity storage system:

### Migration Strategy
1. **Start with in-memory tests**: Create a comprehensive test suite for the in-memory implementation first
2. **Define relationship patterns**: Systematically test all relationship types (one-to-one, one-to-many, many-to-many, hierarchical)
3. **Develop SQLAlchemy backend**: Once in-memory tests pass, implement and test the SQLAlchemy backend against the same test suite
4. **Apply to production models**: Refactor the production models based on proven patterns from the test suite

### Current Progress
We have now implemented both parts of the SQLAlchemy migration:

1. **Entity Storage System**: We've completed the base Entity storage system using SQLAlchemy's ORM with proper UUID handling using SQLAlchemy's built-in `Uuid` type and JSON serialization.

2. **Thread System Models**: We've created a complete set of SQLAlchemy ORM models for all Thread system entities:
   - ChatThread with messages, system prompt, LLM config, and tools
   - ChatMessage with parent-child relationships and usage tracking
   - Tools with polymorphic inheritance (CallableTool and StructuredTool)
   - All other supporting entities (Usage, GeneratedJsonObject, etc.)

The implementation uses:
- Proper SQL table structure with appropriate column types
- Polymorphic inheritance through a discriminator column
- Many-to-many relationships with association tables
- Self-referential relationships for message threading
- Renamed fields to avoid conflicts with SQLAlchemy reserved words

### Testing
We have built a comprehensive test suite that verifies:
1. **Table Structure**: Testing that all tables and columns are created correctly
2. **Basic CRUD**: Creating, querying, and modifying entities
3. **Relationship Loading**: Loading complex entity graphs with eager loading
4. **Many-to-Many Associations**: Thread-to-Tool associations with proper loading
5. **Polymorphic Inheritance**: Tool hierarchy with specialized fields

### Recent Fixes and Improvements

1. **Fixed Field Naming Issues**:
   - Changed field names to match the domain entities (e.g., `description` → `docstring`, `parameters_schema` → `input_schema`)
   - Ensured consistent field naming across entity-to-SQL conversion methods
   - Added proper discriminator fields for polymorphic inheritance

2. **Type Handling Improvements**:
   - Added proper Float types for numeric fields
   - Fixed SQLite UUID handling
   - Improved enum value handling in conversions
   - Implemented proper type conversion for datetime fields with timezone awareness

3. **Relationship Management**:
   - Enhanced relationship handling methods to properly set up bidirectional relationships
   - Added proper session and orm_objects parameters to relationship methods
   - Fixed the entity identity preservation in nested relationships

4. **Test Suite Enhancements**:
   - Updated test helper functions to create valid entities for testing
   - Improved test assertions to check proper field mapping
   - Created dedicated entity identity preservation tests

### Next Steps
1. Integrate the SQLAlchemy models with the EntityRegistry to enable storage and retrieval through the existing API
2. Add SQL-specific storage operations for optimized queries
3. Create migration tools to convert existing SQLModel-based data
4. Update existing tests to work with both storage backends

### Test Location
Tests are being developed in the `/tests/ecs_tests/` directory with:
- `DESIGN.md`: Detailed test planning document
- `conftest.py`: Test fixtures and mock entity classes 
- `test_basic_operations.py`: Tests for basic entity functionality
- `test_relationships.py`: Tests for entity relationships and change propagation

### SQLAlchemy Implementation
The SQLAlchemy implementation in `/tests/sql/sql_entity.py` provides a clean replacement for the current SQLModel-based approach with these key features:

1. **Pure SQLAlchemy Base Classes**:
   - `Base`: Standard SQLAlchemy declarative base
   - `EntityBase`: Abstract base with common entity fields
   - `BaseEntitySQL`: Fallback table for generic entities

2. **Type-Safe Relationship Handling**:
   - Association tables for many-to-many relationships
   - Proper relationship declarations with SQLAlchemy's relationship() function
   - Clear separation between entity and database models

3. **Identical API Behavior**:
   - The SQLAlchemy implementation maintains the same API as the in-memory storage
   - All methods have the same signatures and behaviors
   - This ensures a seamless transition for existing code

4. **Entity Conversion**:
   - `to_entity()`: Converts from SQLAlchemy models to Pydantic entities
   - `from_entity()`: Converts from Pydantic entities to SQLAlchemy models
   - Proper type annotation and casting throughout

5. **Performance Optimizations**:
   - Efficient querying with proper join conditions
   - Session reuse for related operations
   - Two-phase entity storage (entities first, then relationships)

The implementation is structured to maintain identical behavior to the in-memory storage while providing the benefits of SQLAlchemy's robust ORM capabilities. This ensures that the entity versioning system works consistently regardless of which storage backend is in use.

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