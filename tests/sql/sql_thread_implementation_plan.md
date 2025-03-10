# SQLAlchemy Implementation Plan for Thread System

## 1. Create Basic Model Structure

First, I'll create a file structure for our SQL implementation:

```
tests/sql/
├── alchemy_models.py      # Main file with SQLAlchemy models
├── test_thread_sql.py     # Tests for thread system with SQL storage
└── test_thread_entity_conversion.py  # Tests for entity conversion
```

## 2. Implement Core SQLAlchemy Models

### Entity Base Classes

Start with core entity base classes following the pattern we established:

```python
# EntityBase and BaseEntitySQL models from our existing code
# Provides common columns like ecs_id, lineage_id, parent_id, etc.
```

### Core Entity Models

Implement the main entity models that will map to our Pydantic entities:

```python
class ChatThreadSQL(EntityBase):
    """SQLAlchemy model for ChatThread entity."""
    __tablename__ = "chat_thread"
    
    name = mapped_column(String)
    new_message = mapped_column(String, nullable=True)
    prefill = mapped_column(String)
    postfill = mapped_column(String)
    use_schema_instruction = mapped_column(Boolean)
    use_history = mapped_column(Boolean)
    workflow_step = mapped_column(Integer, nullable=True)
    
    # Relationships
    system_prompt_id = mapped_column(ForeignKey("system_prompt.id"), nullable=True)
    llm_config_id = mapped_column(ForeignKey("llm_config.id"))
    forced_output_id = mapped_column(ForeignKey("base_tool.id"), nullable=True)
    
    # Mapped relationships
    system_prompt = relationship("SystemPromptSQL", lazy="joined")
    llm_config = relationship("LLMConfigSQL", lazy="joined") 
    history = relationship("ChatMessageSQL", back_populates="chat_thread", order_by="ChatMessageSQL.timestamp")
    forced_output = relationship("BaseToolSQL", foreign_keys=[forced_output_id])
    tools = relationship("BaseToolSQL", secondary="chat_thread_tools")
    
    # Conversion methods
    def to_entity(self) -> ChatThread:
        """Convert to Pydantic entity."""
        # Implementation
        
    @classmethod
    def from_entity(cls, entity: ChatThread) -> 'ChatThreadSQL':
        """Convert from Pydantic entity."""
        # Implementation
```

### Message and Configuration Models

```python
class ChatMessageSQL(EntityBase):
    """SQLAlchemy model for ChatMessage entity."""
    __tablename__ = "chat_message"
    
    timestamp = mapped_column(DateTime(timezone=True))
    role = mapped_column(String)  # Enum as string
    content = mapped_column(Text)
    author_uuid = mapped_column(SQLAUuid, nullable=True)
    parent_message_uuid = mapped_column(SQLAUuid, nullable=True)
    tool_name = mapped_column(String, nullable=True)
    tool_uuid = mapped_column(SQLAUuid, nullable=True)
    tool_type = mapped_column(String, nullable=True)  # Enum as string
    oai_tool_call_id = mapped_column(String, nullable=True)
    tool_json_schema = mapped_column(JSON, nullable=True)
    tool_call = mapped_column(JSON, nullable=True)
    
    # Foreign keys
    chat_thread_id = mapped_column(ForeignKey("chat_thread.id"))
    usage_id = mapped_column(ForeignKey("usage.id"), nullable=True)
    
    # Relationships
    chat_thread = relationship("ChatThreadSQL", back_populates="history")
    usage = relationship("UsageSQL")
    
    # Conversion methods
    def to_entity(self) -> ChatMessage:
        """Convert to Pydantic entity."""
        # Implementation
        
    @classmethod
    def from_entity(cls, entity: ChatMessage) -> 'ChatMessageSQL':
        """Convert from Pydantic entity."""
        # Implementation
```

### Tool Models with Inheritance

```python
class BaseToolSQL(EntityBase):
    """Base SQLAlchemy model for all tools."""
    __tablename__ = "base_tool"
    
    name = mapped_column(String)
    description = mapped_column(Text, nullable=True)
    tool_type = mapped_column(String)  # Discriminator column
    
    __mapper_args__ = {
        "polymorphic_on": tool_type,
        "polymorphic_identity": "base"
    }
    
    # Conversion methods (abstract)
    def to_entity(self) -> Union[CallableTool, StructuredTool]:
        """Convert to appropriate Pydantic entity."""
        raise NotImplementedError("Must be implemented by subclasses")
        
    @classmethod
    def from_entity(cls, entity: Union[CallableTool, StructuredTool]) -> 'BaseToolSQL':
        """Create appropriate SQLAlchemy model from entity."""
        if isinstance(entity, CallableTool):
            return CallableToolSQL.from_entity(entity)
        elif isinstance(entity, StructuredTool):
            return StructuredToolSQL.from_entity(entity)
        else:
            raise ValueError(f"Unsupported entity type: {type(entity)}")

class CallableToolSQL(BaseToolSQL):
    """SQLAlchemy model for CallableTool entity."""
    __tablename__ = "callable_tool"
    
    id = mapped_column(ForeignKey("base_tool.id"), primary_key=True)
    docstring = mapped_column(Text, nullable=True)
    input_schema = mapped_column(JSON)
    output_schema = mapped_column(JSON)
    strict_schema = mapped_column(Boolean)
    callable_text = mapped_column(Text, nullable=True)
    
    __mapper_args__ = {
        "polymorphic_identity": "callable"
    }
    
    # Conversion methods
    def to_entity(self) -> CallableTool:
        """Convert to Pydantic entity."""
        # Implementation
        
    @classmethod
    def from_entity(cls, entity: CallableTool) -> 'CallableToolSQL':
        """Convert from Pydantic entity."""
        # Implementation

class StructuredToolSQL(BaseToolSQL):
    """SQLAlchemy model for StructuredTool entity."""
    __tablename__ = "structured_tool"
    
    id = mapped_column(ForeignKey("base_tool.id"), primary_key=True)
    instruction_string = mapped_column(Text)
    json_schema = mapped_column(JSON)
    strict_schema = mapped_column(Boolean)
    
    __mapper_args__ = {
        "polymorphic_identity": "structured"
    }
    
    # Conversion methods
    def to_entity(self) -> StructuredTool:
        """Convert to Pydantic entity."""
        # Implementation
        
    @classmethod
    def from_entity(cls, entity: StructuredTool) -> 'StructuredToolSQL':
        """Convert from Pydantic entity."""
        # Implementation
```

### Association Tables

```python
# Many-to-many relationship between ChatThread and Tools
chat_thread_tools = Table(
    "chat_thread_tools",
    Base.metadata,
    Column("chat_thread_id", Integer, ForeignKey("chat_thread.id"), primary_key=True),
    Column("tool_id", Integer, ForeignKey("base_tool.id"), primary_key=True)
)
```

## 3. Implement Conversion Methods

For each SQLAlchemy model, implement comprehensive conversion methods:

```python
# Example for ChatThread
def to_entity(self) -> ChatThread:
    """Convert SQLAlchemy model to Pydantic entity."""
    # Convert system_prompt if present
    system_prompt = self.system_prompt.to_entity() if self.system_prompt else None
    
    # Convert LLM config
    llm_config = self.llm_config.to_entity()
    
    # Convert history
    history = [msg.to_entity() for msg in self.history]
    
    # Convert forced_output and tools
    forced_output = self.forced_output.to_entity() if self.forced_output else None
    tools = [tool.to_entity() for tool in self.tools]
    
    return ChatThread(
        ecs_id=self.ecs_id,
        lineage_id=self.lineage_id,
        parent_id=self.parent_id,
        created_at=self.created_at,
        old_ids=self.old_ids,
        name=self.name,
        system_prompt=system_prompt,
        history=history,
        new_message=self.new_message,
        prefill=self.prefill,
        postfill=self.postfill,
        use_schema_instruction=self.use_schema_instruction,
        use_history=self.use_history,
        forced_output=forced_output,
        llm_config=llm_config,
        tools=tools,
        workflow_step=self.workflow_step
    )

@classmethod
def from_entity(cls, entity: ChatThread) -> 'ChatThreadSQL':
    """Convert Pydantic entity to SQLAlchemy model."""
    # Create the chat thread SQL object
    thread_sql = cls(
        ecs_id=entity.ecs_id,
        lineage_id=entity.lineage_id,
        parent_id=entity.parent_id,
        created_at=entity.created_at,
        old_ids=entity.old_ids,
        name=entity.name,
        new_message=entity.new_message,
        prefill=entity.prefill,
        postfill=entity.postfill,
        use_schema_instruction=entity.use_schema_instruction,
        use_history=entity.use_history,
        workflow_step=entity.workflow_step
    )
    
    # Relationships will be handled by SqlEntityStorage.handle_relationships
    
    return thread_sql

def handle_relationships(self, entity: ChatThread, session: Session, orm_objects: Dict[UUID, Any]) -> None:
    """Handle entity relationships for this model."""
    # Handle system prompt relationship
    if entity.system_prompt:
        if entity.system_prompt.ecs_id in orm_objects:
            self.system_prompt = cast(SystemPromptSQL, orm_objects[entity.system_prompt.ecs_id])
        else:
            self.system_prompt = session.query(SystemPromptSQL).filter(
                SystemPromptSQL.ecs_id == entity.system_prompt.ecs_id
            ).first()
    
    # Handle LLM config
    if entity.llm_config.ecs_id in orm_objects:
        self.llm_config = cast(LLMConfigSQL, orm_objects[entity.llm_config.ecs_id])
    else:
        self.llm_config = session.query(LLMConfigSQL).filter(
            LLMConfigSQL.ecs_id == entity.llm_config.ecs_id
        ).first()
    
    # Handle forced_output
    if entity.forced_output:
        if entity.forced_output.ecs_id in orm_objects:
            self.forced_output = cast(BaseToolSQL, orm_objects[entity.forced_output.ecs_id])
        else:
            self.forced_output = session.query(BaseToolSQL).filter(
                BaseToolSQL.ecs_id == entity.forced_output.ecs_id
            ).first()
    
    # Handle tools (many-to-many)
    self.tools = []
    for tool in entity.tools:
        if tool.ecs_id in orm_objects:
            self.tools.append(cast(BaseToolSQL, orm_objects[tool.ecs_id]))
        else:
            tool_orm = session.query(BaseToolSQL).filter(
                BaseToolSQL.ecs_id == tool.ecs_id
            ).first()
            if tool_orm:
                self.tools.append(tool_orm)
```

## 4. Test Implementation

### Implement Entity Conversion Tests

```python
# test_thread_entity_conversion.py
def test_chat_thread_conversion():
    """Test ChatThread entity-ORM conversion roundtrip."""
    # Create a chat thread with all relationships
    thread = create_sample_chat_thread()
    
    # Convert to ORM
    thread_orm = ChatThreadSQL.from_entity(thread)
    
    # Save to database
    session.add(thread_orm)
    session.commit()
    
    # Retrieve from database
    retrieved_orm = session.query(ChatThreadSQL).filter(
        ChatThreadSQL.ecs_id == thread.ecs_id
    ).first()
    
    # Convert back to entity
    retrieved_thread = retrieved_orm.to_entity()
    
    # Check equality
    assert compare_entities(thread, retrieved_thread)
```

### Implement Modification Detection Tests

```python
# test_thread_sql.py
async def test_entity_tracer_with_sql_storage():
    """Test that entity_tracer works with SQL storage."""
    # Initialize SQL storage
    sql_storage = SqlEntityStorage(
        session_factory=get_session,
        entity_to_orm_map=entity_to_orm_map
    )
    
    # Set up EntityRegistry to use SQL storage
    EntityRegistry.use_storage(sql_storage)
    
    # Create and register a thread
    thread = create_sample_chat_thread()
    EntityRegistry.register(thread)
    
    # Retrieve thread from storage
    retrieved_thread = EntityRegistry.get(thread.ecs_id)
    
    # Modify thread and check version change
    retrieved_thread.new_message = "Hello, world!"
    await retrieved_thread.add_user_message()
    
    # Check that a new version was created
    assert retrieved_thread.ecs_id != thread.ecs_id
    assert retrieved_thread.parent_id == thread.ecs_id
```

### Test Full Thread Operations with SQL Storage

```python
# test_thread_sql.py
async def test_full_thread_interaction():
    """Test a complete thread interaction flow with SQL storage."""
    # Initialize SQL storage
    sql_storage = SqlEntityStorage(
        session_factory=get_session,
        entity_to_orm_map=entity_to_orm_map
    )
    
    # Set up EntityRegistry to use SQL storage
    EntityRegistry.use_storage(sql_storage)
    
    # Create thread with tool
    thread = create_sample_chat_thread_with_tool()
    EntityRegistry.register(thread)
    
    # Add user message
    thread.new_message = "Calculate 2+2"
    await thread.add_user_message()
    
    # Mock LLM call and response
    raw_output = create_mock_raw_output(thread)
    processed_output = raw_output.create_processed_output()
    
    # Add chat turn with tool execution
    await thread.add_chat_turn_history(processed_output)
    
    # Verify chat history structure
    assert len(thread.history) == 3  # User, assistant, tool messages
    assert thread.history[1].role == MessageRole.assistant
    assert thread.history[2].role == MessageRole.tool
    
    # Verify tool execution records
    assert thread.history[1].tool_call is not None
    assert thread.history[2].tool_name == "calculator"
```

## 5. Integration with Existing Tests

Modify the existing threads test suite to work with SQL storage:

```python
# tests/threads_tests/conftest.py modification
@pytest.fixture
def use_sql_storage():
    """Configure EntityRegistry to use SQL storage."""
    # Initialize in-memory SQLite
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    
    # Create entity-to-ORM mapping
    entity_to_orm_map = {
        ChatThread: ChatThreadSQL,
        ChatMessage: ChatMessageSQL,
        CallableTool: CallableToolSQL,
        StructuredTool: StructuredToolSQL,
        LLMConfig: LLMConfigSQL,
        SystemPrompt: SystemPromptSQL,
        Usage: UsageSQL,
        GeneratedJsonObject: GeneratedJsonObjectSQL,
        RawOutput: RawOutputSQL,
        ProcessedOutput: ProcessedOutputSQL,
        Entity: BaseEntitySQL  # Fallback
    }
    
    # Create SQL storage
    sql_storage = SqlEntityStorage(
        session_factory=Session,
        entity_to_orm_map=entity_to_orm_map
    )
    
    # Save original storage
    original_storage = EntityRegistry._storage
    
    # Set EntityRegistry to use SQL storage
    EntityRegistry.use_storage(sql_storage)
    
    yield sql_storage
    
    # Restore original storage
    EntityRegistry.use_storage(original_storage)
```

Then modify test cases to use this fixture:

```python
# Existing tests with SQL storage fixture
def test_chat_thread_message_adding(use_sql_storage):
    # Test remains mostly unchanged, but now uses SQL storage
    thread = ChatThread(...)
    thread.add_user_message()
    ...
```

## Timeline and Approach

1. **First Implementation Phase (1-2 days)**
   - Create all SQLAlchemy models with basic conversion methods
   - Implement simple entity conversion tests
   - Focus on the core entities first: ChatThread, ChatMessage, LLMConfig

2. **Middle Implementation Phase (2-3 days)**
   - Add tool inheritance structure
   - Implement relationship handling
   - Test more complex entity structures with relationships

3. **Final Implementation Phase (2-3 days)**
   - Convert existing tests to use SQL storage
   - Implement modification detection tests
   - Test full thread operations with tool execution

4. **Optimization and Cleanup (1-2 days)**
   - Optimize queries for performance
   - Add indexes for common query patterns
   - Improve error handling and edge cases

This implementation will provide a solid foundation for storing and retrieving thread entities in a SQL database while maintaining all the versioning and relationship features of the current in-memory implementation.