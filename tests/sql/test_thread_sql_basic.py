"""
Basic tests for the SQL thread models.

These tests verify that the SQL models can be created and saved to the database.
"""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from minference.threads.sql_models import (
    Base, ChatThreadSQL, ChatMessageSQL, SystemPromptSQL, LLMConfigSQL, 
    CallableToolSQL, StructuredToolSQL, UsageSQL
)

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

def test_create_and_query_system_prompt(session):
    """Test creating and querying a SystemPrompt."""
    # Create a system prompt
    system_prompt = SystemPromptSQL(
        ecs_id=uuid.uuid4(),
        lineage_id=uuid.uuid4(),
        parent_id=None,
        created_at=datetime.now(timezone.utc),
        old_ids=[],
        entity_type="system_prompt",
        content="You are a helpful assistant.",
        prompt_name="Default",
        prompt_description="A default system prompt"
    )
    
    # Save to database
    session.add(system_prompt)
    session.commit()
    
    # Query from database
    result = session.execute(select(SystemPromptSQL)).scalar_one()
    
    # Check fields
    assert result.ecs_id == system_prompt.ecs_id
    assert result.content == "You are a helpful assistant."
    assert result.prompt_name == "Default"

def test_create_and_query_chat_thread_with_messages_and_tools(session):
    """Test creating and querying a ChatThread with messages, system prompt, LLM config, and tools."""
    # Create a system prompt
    system_prompt = SystemPromptSQL(
        ecs_id=uuid.uuid4(),
        lineage_id=uuid.uuid4(),
        parent_id=None,
        created_at=datetime.now(timezone.utc),
        old_ids=[],
        entity_type="system_prompt",
        content="You are a helpful assistant.",
        prompt_name="Default",
        prompt_description=None
    )
    
    # Create an LLM config
    llm_config = LLMConfigSQL(
        ecs_id=uuid.uuid4(),
        lineage_id=uuid.uuid4(),
        parent_id=None,
        created_at=datetime.now(timezone.utc),
        old_ids=[],
        entity_type="llm_config",
        model="gpt-4",
        provider_name="openai",
        provider_api_key=None,
        max_tokens=1000,
        temperature=0.7,
        response_format={"type": "text"},
        llm_config=None
    )
    
    # Create some tools
    callable_tool = CallableToolSQL(
        ecs_id=uuid.uuid4(),
        lineage_id=uuid.uuid4(),
        parent_id=None,
        created_at=datetime.now(timezone.utc),
        old_ids=[],
        entity_type="callable_tool",
        tool_type="callable_tool",  # Polymorphic discriminator
        name="get_weather",
        tool_description="Get weather forecast for a location",
        tool_parameters_schema={"type": "object", "properties": {"location": {"type": "string"}}},
        input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"temperature": {"type": "number"}}},
        strict_schema=True,
        callable_text="def get_weather(location: str) -> dict:\n    return {'temperature': 72}"
    )
    
    structured_tool = StructuredToolSQL(
        ecs_id=uuid.uuid4(),
        lineage_id=uuid.uuid4(),
        parent_id=None,
        created_at=datetime.now(timezone.utc),
        old_ids=[],
        entity_type="tool",
        tool_type="structured_tool",  # Polymorphic discriminator
        name="search",
        tool_description="Search the internet for information",
        tool_parameters_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        tool_output_schema={"type": "object", "properties": {"results": {"type": "array"}}},
        instruction_string="Please follow this JSON schema for search results",
        strict_schema=True
    )
    
    # Create a chat thread
    chat_thread = ChatThreadSQL(
        ecs_id=uuid.uuid4(),
        lineage_id=uuid.uuid4(),
        parent_id=None,
        created_at=datetime.now(timezone.utc),
        old_ids=[],
        entity_type="chat_thread",
        title="Test Thread",
        thread_metadata={"test": True},
        system_prompt=system_prompt,
        llm_config=llm_config,
        tools=[callable_tool, structured_tool]
    )
    
    # Create a user message
    user_message = ChatMessageSQL(
        ecs_id=uuid.uuid4(),
        lineage_id=uuid.uuid4(),
        parent_id=None,
        created_at=datetime.now(timezone.utc),
        old_ids=[],
        entity_type="chat_message",
        role="user",
        content="Hello, world!",
        chat_thread=chat_thread
    )
    
    # Create an assistant message
    assistant_message = ChatMessageSQL(
        ecs_id=uuid.uuid4(),
        lineage_id=uuid.uuid4(),
        parent_id=None,
        created_at=datetime.now(timezone.utc),
        old_ids=[],
        entity_type="chat_message",
        role="assistant",
        content="Hi there! How can I help you?",
        parent_message=user_message,
        chat_thread=chat_thread
    )
    
    # Save to database
    session.add(chat_thread)
    session.add(user_message)
    session.add(assistant_message)
    session.commit()
    
    # Query from database with eager loading
    from sqlalchemy.orm import joinedload
    result = session.execute(
        select(ChatThreadSQL)
        .options(
            joinedload(ChatThreadSQL.messages),
            joinedload(ChatThreadSQL.system_prompt),
            joinedload(ChatThreadSQL.llm_config),
            joinedload(ChatThreadSQL.tools)
        )
    ).unique().scalar_one()
    
    # Check fields
    assert result.ecs_id == chat_thread.ecs_id
    assert result.title == "Test Thread"
    assert result.system_prompt.content == "You are a helpful assistant."
    assert result.llm_config.model == "gpt-4"
    assert len(result.messages) == 2
    assert len(result.tools) == 2
    
    # Check message types
    roles = [msg.role for msg in result.messages]
    assert "user" in roles
    assert "assistant" in roles
    
    # Check message content
    for msg in result.messages:
        if msg.role == "user":
            assert msg.content == "Hello, world!"
        elif msg.role == "assistant":
            assert msg.content == "Hi there! How can I help you?"
            assert msg.parent_message.role == "user"
    
    # Check tools
    tool_names = [tool.name for tool in result.tools]
    assert "get_weather" in tool_names
    assert "search" in tool_names
    
    # Check tool types
    for tool in result.tools:
        if tool.name == "get_weather":
            assert tool.entity_type == "callable_tool"
            assert tool.tool_type == "callable_tool"
            assert tool.tool_parameters_schema["properties"]["location"]["type"] == "string"
            assert hasattr(tool, "input_schema")
            assert hasattr(tool, "output_schema")
            assert hasattr(tool, "callable_text")
        elif tool.name == "search":
            assert tool.entity_type == "tool"
            assert tool.tool_type == "structured_tool"
            assert tool.tool_output_schema["properties"]["results"]["type"] == "array"
            assert hasattr(tool, "instruction_string")