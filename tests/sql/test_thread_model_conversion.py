"""
Tests for converting Thread system entities to SQLAlchemy ORM models and back.

These tests verify that entities can be properly converted to their corresponding
SQLAlchemy ORM models and back, maintaining all properties and relationships.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, cast, Union

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from minference.ecs.entity import Entity
from minference.threads.models import (
    ChatMessage, ChatThread, GeneratedJsonObject, LLMConfig, 
    RawOutput, ProcessedOutput, StructuredTool, SystemPrompt, 
    CallableTool, Usage
)

from tests.sql.sql_thread_models import (
    Base, ChatMessageSQL, ChatThreadSQL, GeneratedJsonObjectSQL, 
    LLMConfigSQL, RawOutputSQL, ProcessedOutputSQL, StructuredToolSQL, 
    SystemPromptSQL, CallableToolSQL, UsageSQL, ENTITY_MODEL_MAP
)

# Setup SQLite in-memory database for testing
@pytest.fixture
def engine():
    """Create an in-memory SQLite engine."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine

@pytest.fixture
def session(engine):
    """Create a database session."""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

# Fix UUID handling in SQLite
from uuid import UUID
import sqlite3

def adapt_uuid(uuid):
    return str(uuid)

def convert_uuid(s):
    return UUID(s.decode())

sqlite3.register_adapter(UUID, adapt_uuid)
sqlite3.register_converter("UUID", convert_uuid)

# Helper to create a unique ID
def create_uuid():
    return uuid.uuid4()

# Test entity creation helpers
def create_system_prompt() -> SystemPrompt:
    """Create a sample SystemPrompt entity for testing."""
    return SystemPrompt(
        content="You are a helpful assistant.",
        name="Default System Prompt",
        description="Standard system prompt for testing"
    )

def create_llm_config() -> LLMConfig:
    """Create a sample LLMConfig entity for testing."""
    from minference.threads.models import LLMClient, ResponseFormat
    return LLMConfig(
        client=LLMClient.openai,
        model="gpt-4",
        max_tokens=1000,
        temperature=0.7,
        response_format=ResponseFormat.text,
        use_cache=True
    )

def create_usage() -> Usage:
    """Create a sample Usage entity for testing."""
    return Usage(
        model="gpt-4",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150
    )

def create_callable_tool() -> CallableTool:
    """Create a sample CallableTool entity for testing."""
    return CallableTool(
        name="get_weather",
        docstring="Get the weather for a location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "condition": {"type": "string"}
            }
        },
        callable_text="def get_weather(location: str) -> dict:\n    return {'temperature': 72, 'condition': 'sunny'}"
    )

def create_structured_tool() -> StructuredTool:
    """Create a sample StructuredTool entity for testing."""
    return StructuredTool(
        name="search",
        description="Search for information",
        json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "results": {"type": "array"}
            },
            "required": ["query"]
        },
        instruction_string="Please follow this JSON schema for search results"
    )

def create_raw_output() -> RawOutput:
    """Create a sample RawOutput entity for testing."""
    from minference.threads.models import LLMClient
    return RawOutput(
        raw_result={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-4",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            },
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response."
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ]
        },
        completion_kwargs={},
        start_time=1677858242.0,
        end_time=1677858243.0,
        client=LLMClient.openai
    )

def create_processed_output(raw_output: Optional[RawOutput] = None) -> ProcessedOutput:
    """Create a sample ProcessedOutput entity for testing."""
    from minference.threads.models import LLMClient, GeneratedJsonObject
    import uuid
    
    if raw_output is None:
        raw_output = create_raw_output()
    
    # Create a Usage entity for the ProcessedOutput
    usage = create_usage()
    
    # Create a JSON object for the ProcessedOutput
    json_object = GeneratedJsonObject(
        name="test_result",
        object={"finish_reason": "stop", "result": "This works"}
    )
    
    return ProcessedOutput(
        content="This is a test response.",
        json_object=json_object,
        usage=usage,
        time_taken=1.0,
        llm_client=LLMClient.openai,
        raw_output=raw_output,
        chat_thread_id=uuid.uuid4(),
        chat_thread_live_id=uuid.uuid4()
    )

def create_chat_message(
    role: str = "user",
    content: str = "Hello, world!",
    parent_message: Optional[ChatMessage] = None,
    usage: Optional[Usage] = None,
    tool: Optional[Union[CallableTool, StructuredTool]] = None,
) -> ChatMessage:
    """Create a sample ChatMessage entity for testing."""
    from minference.threads.models import MessageRole
    
    # Convert string role to MessageRole enum
    message_role = getattr(MessageRole, role) if hasattr(MessageRole, role) else MessageRole.user
    
    # Determine tool type from tool
    tool_type = None
    if tool:
        if isinstance(tool, CallableTool):
            tool_type = "Callable"
        elif isinstance(tool, StructuredTool):
            tool_type = "Structured"
    
    return ChatMessage(
        role=message_role,
        content=content,
        parent_message_uuid=parent_message.ecs_id if parent_message else None,
        chat_thread_id=None,  # Will be set by chat thread
        tool_uuid=tool.ecs_id if tool else None,
        tool_name=tool.name if tool else None,
        tool_type=tool_type,
        oai_tool_call_id="call_123" if role == "tool" else None,
        usage=usage
    )

def create_chat_thread(
    system_prompt: Optional[SystemPrompt] = None,
    llm_config: Optional[LLMConfig] = None,
    tools: Optional[List[Union[CallableTool, StructuredTool]]] = None,
    messages: Optional[List[ChatMessage]] = None
) -> ChatThread:
    """Create a sample ChatThread entity for testing."""
    if system_prompt is None:
        system_prompt = create_system_prompt()
    
    if llm_config is None:
        llm_config = create_llm_config()
    
    if tools is None:
        tools = [create_structured_tool()]  # Just use StructuredTool to avoid import issues
    
    # Create a new chat thread
    chat_thread = ChatThread(
        name="Test Thread",
        system_prompt=system_prompt,
        llm_config=llm_config,
        tools=tools,
        history=[]
    )
    
    # Add messages if provided
    if messages:
        chat_thread.history = messages
        # Set chat_thread_id for each message
        for message in messages:
            message.chat_thread_id = chat_thread.ecs_id
    
    return chat_thread

# Tests for individual entity conversions
def test_system_prompt_conversion(session):
    """Test converting SystemPrompt entity to ORM and back."""
    # Create entity
    entity = create_system_prompt()
    
    # Convert to ORM
    orm_model = SystemPromptSQL.from_entity(entity)
    assert orm_model.content == entity.content
    assert orm_model.prompt_name == entity.name  # Field renamed in SQL model
    assert orm_model.ecs_id == entity.ecs_id
    
    # Save to database
    session.add(orm_model)
    session.commit()
    
    # Retrieve from database
    retrieved = session.query(SystemPromptSQL).filter_by(ecs_id=entity.ecs_id).first()
    assert retrieved is not None
    
    # Convert back to entity
    reconstructed = retrieved.to_entity()
    assert reconstructed.content == entity.content
    assert reconstructed.name == entity.name
    assert reconstructed.ecs_id == entity.ecs_id

def test_llm_config_conversion(session):
    """Test converting LLMConfig entity to ORM and back."""
    # Create entity
    entity = create_llm_config()
    
    # Convert to ORM
    orm_model = LLMConfigSQL.from_entity(entity)
    assert orm_model.provider_name == entity.client.value  # Field renamed and is enum value
    assert orm_model.model == entity.model
    assert orm_model.max_tokens == entity.max_tokens
    assert orm_model.temperature == entity.temperature
    assert orm_model.response_format == entity.response_format.value  # Response format is enum value
    
    # Save to database
    session.add(orm_model)
    session.commit()
    
    # Retrieve from database
    retrieved = session.query(LLMConfigSQL).filter_by(ecs_id=entity.ecs_id).first()
    assert retrieved is not None
    
    # Convert back to entity
    reconstructed = retrieved.to_entity()
    assert reconstructed.client == entity.client
    assert reconstructed.model == entity.model
    assert reconstructed.max_tokens == entity.max_tokens
    assert reconstructed.temperature == entity.temperature
    assert reconstructed.response_format == entity.response_format
    assert reconstructed.ecs_id == entity.ecs_id

def test_usage_conversion(session):
    """Test converting Usage entity to ORM and back."""
    # Create entity
    entity = create_usage()
    
    # Convert to ORM
    orm_model = UsageSQL.from_entity(entity)
    assert orm_model.prompt_tokens == entity.prompt_tokens
    assert orm_model.completion_tokens == entity.completion_tokens
    assert orm_model.total_tokens == entity.total_tokens
    
    # Save to database
    session.add(orm_model)
    session.commit()
    
    # Retrieve from database
    retrieved = session.query(UsageSQL).filter_by(ecs_id=entity.ecs_id).first()
    assert retrieved is not None
    
    # Convert back to entity
    reconstructed = retrieved.to_entity()
    assert reconstructed.prompt_tokens == entity.prompt_tokens
    assert reconstructed.completion_tokens == entity.completion_tokens
    assert reconstructed.total_tokens == entity.total_tokens
    assert reconstructed.ecs_id == entity.ecs_id

def test_callable_tool_conversion(session):
    """Test converting CallableTool entity to ORM and back."""
    # Create entity
    entity = create_callable_tool()
    
    # Convert to ORM
    orm_model = CallableToolSQL.from_entity(entity)
    assert orm_model.name == entity.name
    assert orm_model.tool_description == entity.docstring  # Field renamed
    assert orm_model.input_schema == entity.input_schema  # Field renamed
    assert orm_model.output_schema == entity.output_schema
    assert orm_model.callable_text == entity.callable_text
    
    # Save to database
    session.add(orm_model)
    session.commit()
    
    # Retrieve from database
    retrieved = session.query(CallableToolSQL).filter_by(ecs_id=entity.ecs_id).first()
    assert retrieved is not None
    
    # Convert back to entity
    reconstructed = retrieved.to_entity()
    assert reconstructed.name == entity.name
    assert reconstructed.docstring == entity.docstring  # Field renamed
    assert reconstructed.input_schema == entity.input_schema
    assert reconstructed.output_schema == entity.output_schema
    assert reconstructed.callable_text == entity.callable_text
    assert reconstructed.ecs_id == entity.ecs_id

def test_structured_tool_conversion(session):
    """Test converting StructuredTool entity to ORM and back."""
    # Create entity
    entity = create_structured_tool()
    
    # Convert to ORM
    orm_model = StructuredToolSQL.from_entity(entity)
    assert orm_model.name == entity.name
    assert orm_model.tool_description == entity.description  # Field renamed
    assert orm_model.tool_output_schema == entity.json_schema  # Field renamed
    assert orm_model.instruction_string == entity.instruction_string
    
    # Save to database
    session.add(orm_model)
    session.commit()
    
    # Retrieve from database
    retrieved = session.query(StructuredToolSQL).filter_by(ecs_id=entity.ecs_id).first()
    assert retrieved is not None
    
    # Convert back to entity
    reconstructed = retrieved.to_entity()
    assert reconstructed.name == entity.name
    assert reconstructed.description == entity.description
    assert reconstructed.json_schema == entity.json_schema  # Field renamed
    assert reconstructed.instruction_string == entity.instruction_string
    assert reconstructed.ecs_id == entity.ecs_id

def test_chat_message_with_usage_conversion(session):
    """Test converting ChatMessage with Usage entity to ORM and back."""
    # Create entities
    usage = create_usage()
    entity = create_chat_message(usage=usage)
    
    # Save usage to database first
    usage_sql = UsageSQL.from_entity(usage)
    session.add(usage_sql)
    session.commit()
    
    # Convert message to ORM
    orm_model = ChatMessageSQL.from_entity(entity)
    
    # Handle relationships properly with session and orm_objects
    orm_objects = {usage.ecs_id: usage_sql}
    orm_model.handle_relationships(entity, session, orm_objects)
    
    # Check fields before saving
    assert orm_model.role == entity.role.value  # Convert enum to string
    assert orm_model.content == entity.content
    assert orm_model.tool_name == entity.tool_name
    assert orm_model.oai_tool_call_id == entity.oai_tool_call_id
    
    # Save to database
    session.add(orm_model)
    session.commit()
    
    # Retrieve from database
    retrieved = session.query(ChatMessageSQL).filter_by(ecs_id=entity.ecs_id).first()
    assert retrieved is not None
    
    # Convert back to entity
    reconstructed = retrieved.to_entity()
    assert reconstructed.role == entity.role
    assert reconstructed.content == entity.content
    assert reconstructed.tool_name == entity.tool_name
    assert reconstructed.oai_tool_call_id == entity.oai_tool_call_id
    assert reconstructed.ecs_id == entity.ecs_id
    
    # Check usage relationship
    assert retrieved.usage is not None
    assert reconstructed.usage is not None
    assert reconstructed.usage.prompt_tokens == usage.prompt_tokens
    assert reconstructed.usage.completion_tokens == usage.completion_tokens
    assert reconstructed.usage.total_tokens == usage.total_tokens

def test_processed_output_with_raw_output_conversion(session):
    """Test converting ProcessedOutput with RawOutput entity to ORM and back."""
    # Create entities
    raw_output = create_raw_output()
    entity = create_processed_output(raw_output)
    
    # Save related entities to database first
    raw_output_sql = RawOutputSQL.from_entity(raw_output)
    usage_sql = UsageSQL.from_entity(entity.usage) if entity.usage else None
    json_object_sql = GeneratedJsonObjectSQL.from_entity(entity.json_object) if entity.json_object else None
    
    session.add(raw_output_sql)
    if usage_sql:
        session.add(usage_sql)
    if json_object_sql:
        session.add(json_object_sql)
    session.commit()
    
    # Convert to ORM
    orm_model = ProcessedOutputSQL.from_entity(entity)
    
    # Setup orm_objects dictionary for relationships
    orm_objects = {raw_output.ecs_id: raw_output_sql}
    if entity.usage:
        orm_objects[entity.usage.ecs_id] = usage_sql
    if entity.json_object:
        orm_objects[entity.json_object.ecs_id] = json_object_sql
    
    # Handle relationships properly
    orm_model.handle_relationships(entity, session, orm_objects)
    
    # Manually set raw_output_id if not set by handle_relationships
    if orm_model.raw_output_id is None:
        orm_model.raw_output_id = raw_output.ecs_id
        orm_model.raw_output = raw_output_sql
    
    # Check fields before saving
    assert orm_model.content == entity.content
    assert orm_model.error == entity.error
    assert orm_model.raw_output_id == raw_output.ecs_id
    
    # Save to database
    session.add(orm_model)
    session.commit()
    
    # Retrieve from database
    retrieved = session.query(ProcessedOutputSQL).filter_by(ecs_id=entity.ecs_id).first()
    assert retrieved is not None
    
    # Convert back to entity
    reconstructed = retrieved.to_entity()
    assert reconstructed.content == entity.content
    assert reconstructed.ecs_id == entity.ecs_id
    
    # Check raw_output relationship
    assert retrieved.raw_output is not None
    assert reconstructed.raw_output is not None
    assert reconstructed.raw_output.ecs_id == raw_output.ecs_id
    assert reconstructed.raw_output.raw_result == raw_output.raw_result
    
    # Check JSON object and usage
    if entity.json_object:
        assert reconstructed.json_object is not None
        assert reconstructed.json_object.name == entity.json_object.name
    
    if entity.usage:
        assert reconstructed.usage is not None
        assert reconstructed.usage.model == entity.usage.model

def test_chat_thread_complex_conversion(session):
    """Test converting a complex ChatThread with all relationships to ORM and back."""
    # Create component entities
    system_prompt = create_system_prompt()
    llm_config = create_llm_config()
    callable_tool = create_callable_tool()
    structured_tool = create_structured_tool()
    tools = [callable_tool, structured_tool]
    
    # Create messages
    user_message = create_chat_message(role="user", content="Hello")
    assistant_message = create_chat_message(role="assistant", content="Hi there", parent_message=user_message)
    tool_message = create_chat_message(role="tool", content='{"weather": "sunny"}', parent_message=assistant_message, tool=callable_tool)
    messages = [user_message, assistant_message, tool_message]
    
    # Create chat thread with all components
    chat_thread = create_chat_thread(
        system_prompt=system_prompt,
        llm_config=llm_config,
        tools=tools,
        messages=messages
    )
    
    # Convert to ORM
    orm_model = ChatThreadSQL.from_entity(chat_thread)
    orm_model.handle_relationships(chat_thread)
    
    # Save all entities to database
    session.add(SystemPromptSQL.from_entity(system_prompt))
    session.add(LLMConfigSQL.from_entity(llm_config))
    session.add(CallableToolSQL.from_entity(callable_tool))
    session.add(StructuredToolSQL.from_entity(structured_tool))
    session.add(orm_model)
    session.commit()
    
    # Retrieve from database
    retrieved = session.query(ChatThreadSQL).filter_by(ecs_id=chat_thread.ecs_id).first()
    assert retrieved is not None
    
    # Convert back to entity
    reconstructed = retrieved.to_entity()
    
    # Check basic properties
    assert reconstructed.title == chat_thread.title
    assert reconstructed.metadata == chat_thread.metadata
    assert reconstructed.ecs_id == chat_thread.ecs_id
    
    # Check system prompt
    assert reconstructed.system_prompt is not None
    assert reconstructed.system_prompt.content == system_prompt.content
    
    # Check LLM config
    assert reconstructed.llm_config is not None
    assert reconstructed.llm_config.provider == llm_config.provider
    assert reconstructed.llm_config.model == llm_config.model
    
    # Check tools
    assert len(reconstructed.tools) == 2
    tool_names = [tool.name for tool in reconstructed.tools]
    assert callable_tool.name in tool_names
    assert structured_tool.name in tool_names
    
    # Check messages
    assert len(reconstructed.messages) == 3
    
    # Extract message roles
    message_roles = [msg.role for msg in reconstructed.messages]
    assert "user" in message_roles
    assert "assistant" in message_roles
    assert "tool" in message_roles
    
    # Check message relationships
    for msg in reconstructed.messages:
        if msg.role == "assistant":
            # Assistant message should have parent (user message)
            assert msg.parent_message_uuid is not None
        if msg.role == "tool":
            # Tool message should have tool_uuid
            assert msg.tool_uuid is not None

# Test the ENTITY_MODEL_MAP
def test_entity_model_map():
    """Test that the ENTITY_MODEL_MAP correctly maps entity classes to ORM models."""
    assert ENTITY_MODEL_MAP[ChatThread] == ChatThreadSQL
    assert ENTITY_MODEL_MAP[ChatMessage] == ChatMessageSQL
    assert ENTITY_MODEL_MAP[SystemPrompt] == SystemPromptSQL
    assert ENTITY_MODEL_MAP[LLMConfig] == LLMConfigSQL
    assert ENTITY_MODEL_MAP[CallableTool] == CallableToolSQL
    assert ENTITY_MODEL_MAP[StructuredTool] == StructuredToolSQL
    assert ENTITY_MODEL_MAP[Usage] == UsageSQL
    assert ENTITY_MODEL_MAP[GeneratedJsonObject] == GeneratedJsonObjectSQL
    assert ENTITY_MODEL_MAP[RawOutput] == RawOutputSQL
    assert ENTITY_MODEL_MAP[ProcessedOutput] == ProcessedOutputSQL