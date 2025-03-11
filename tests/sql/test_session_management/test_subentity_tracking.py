"""
Tests for sub-entity tracking in SQL storage.

This verifies that all sub-entities are properly found and registered
when registering a parent entity with the SQL storage backend.
"""

import sys
import pytest
import uuid
from uuid import UUID
from typing import List, Dict, Any, Optional, cast

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from minference.ecs.entity import Entity
from minference.ecs.enregistry import EntityRegistry, entity_tracer
from minference.ecs.storage import SqlEntityStorage
from minference.threads.models import (
    ChatThread, ChatMessage, LLMConfig, LLMClient, MessageRole,
    ResponseFormat, CallableTool, StructuredTool, SystemPrompt
)
from minference.threads.sql_models import ENTITY_MODEL_MAP


@pytest.fixture(scope="function")
def session_factory():
    """Create an in-memory SQLite database and session factory."""
    # Import all needed models to ensure tables are created
    from minference.threads.sql_models import (
        Base, ChatThreadSQL, ChatMessageSQL, LLMConfigSQL, 
        SystemPromptSQL, ToolSQL, CallableToolSQL, StructuredToolSQL,
        UsageSQL, GeneratedJsonObjectSQL, RawOutputSQL, ProcessedOutputSQL
    )
    from minference.ecs.storage import BaseEntitySQL
    from sqlalchemy import text
    
    # Create engine for SQLite in-memory DB with foreign keys enabled
    engine = create_engine("sqlite:///:memory:", echo=False)
    
    # Explicitly create the base_entity table
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE base_entity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ecs_id CHAR(32) NOT NULL,
            lineage_id CHAR(32) NOT NULL,
            parent_id CHAR(32),
            created_at DATETIME NOT NULL,
            old_ids JSON NOT NULL,
            class_name VARCHAR(255) NOT NULL,
            data JSON,
            entity_type VARCHAR(50) NOT NULL
        )
        """))
        conn.commit()
    
    # Create all other tables
    Base.metadata.create_all(engine)
    
    # Create a session factory
    return sessionmaker(bind=engine)


@pytest.fixture
def setup_sql_storage(session_factory):
    """Set up SQL storage for the EntityRegistry."""
    # Add EntityRegistry to __main__ for entity methods
    sys.modules['__main__'].__dict__['EntityRegistry'] = EntityRegistry
    
    # Create SQL storage with the session factory
    sql_storage = SqlEntityStorage(
        session_factory=session_factory,
        entity_to_orm_map=ENTITY_MODEL_MAP
    )
    
    # Save original storage to restore later
    original_storage = EntityRegistry._storage
    
    # Set SQL storage for testing
    EntityRegistry.use_storage(sql_storage)
    
    yield
    
    # Restore original storage
    EntityRegistry._storage = original_storage


def test_complex_nested_entities(setup_sql_storage):
    """Test registration of a complex structure with multiple levels of nested entities."""
    
    # Create a thread with system prompt, LLM config, and tools
    system_prompt = SystemPrompt(
        name="Test Prompt",
        content="You are a helpful assistant."
    )
    
    llm_config = LLMConfig(
        client=LLMClient.openai,
        model="gpt-4",
        response_format=ResponseFormat.json_object,
        temperature=0.7
    )
    
    # Create a callable tool
    calc_tool = CallableTool(
        name="calculator",
        docstring="Performs simple calculations",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "operation": {"type": "string", "enum": ["+", "-", "*", "/"]}
            },
            "required": ["a", "b", "operation"]
        }
    )
    
    # Create a structured tool
    json_tool = StructuredTool(
        name="json_formatter",
        description="Formats JSON data nicely",
        json_schema={
            "type": "object",
            "properties": {
                "formatted": {"type": "boolean"},
                "data": {"type": "object"}
            }
        }
    )
    
    # Create a thread with all the above components
    thread = ChatThread(
        name="Complex Thread",
        system_prompt=system_prompt,
        llm_config=llm_config,
        tools=[calc_tool, json_tool]
    )
    
    # Add some messages to the thread
    thread.new_message = "Hello, I need to calculate 2+2"
    msg1 = thread.add_user_message()
    
    thread.new_message = "I'll help you calculate that!"
    msg2 = ChatMessage(
        role=MessageRole.assistant,
        content="I'll help you calculate that!",
        chat_thread_id=thread.ecs_id,
        parent_message_uuid=msg1.ecs_id
    )
    thread.history.append(msg2)
    
    # Register only the thread - all nested entities should be registered automatically
    registered_thread = EntityRegistry.register(thread)
    
    # Verify the thread was registered
    assert registered_thread is not None
    assert registered_thread.ecs_id == thread.ecs_id
    
    # Verify system prompt was registered
    retrieved_prompt = SystemPrompt.get(system_prompt.ecs_id)
    assert retrieved_prompt is not None
    assert retrieved_prompt.content == "You are a helpful assistant."
    
    # Verify LLM config was registered
    retrieved_config = LLMConfig.get(llm_config.ecs_id)
    assert retrieved_config is not None
    assert retrieved_config.model == "gpt-4"
    assert retrieved_config.temperature == 0.7
    
    # Verify tools were registered
    retrieved_calc_tool = CallableTool.get(calc_tool.ecs_id)
    assert retrieved_calc_tool is not None
    assert retrieved_calc_tool.name == "calculator"
    
    retrieved_json_tool = StructuredTool.get(json_tool.ecs_id)
    assert retrieved_json_tool is not None
    assert retrieved_json_tool.name == "json_formatter"
    
    # Verify messages were registered
    retrieved_msg1 = ChatMessage.get(msg1.ecs_id)
    assert retrieved_msg1 is not None
    assert retrieved_msg1.content == "Hello, I need to calculate 2+2"
    
    retrieved_msg2 = ChatMessage.get(msg2.ecs_id)
    assert retrieved_msg2 is not None
    assert retrieved_msg2.content == "I'll help you calculate that!"
    
    # Now verify the complete thread with all its relationships
    complete_thread = ChatThread.get(registered_thread.ecs_id)
    assert complete_thread is not None
    assert complete_thread.name == "Complex Thread"
    assert complete_thread.system_prompt is not None
    assert complete_thread.system_prompt.ecs_id == system_prompt.ecs_id
    assert complete_thread.llm_config is not None
    assert complete_thread.llm_config.ecs_id == llm_config.ecs_id
    assert len(complete_thread.tools) == 2
    assert len(complete_thread.history) == 2
    
    # Verify parent-child relationships in messages
    messages_by_id = {msg.ecs_id: msg for msg in complete_thread.history}
    assert retrieved_msg2.parent_message_uuid == retrieved_msg1.ecs_id
    

@entity_tracer
def create_thread_with_messages() -> ChatThread:
    """Create a thread with messages using the entity_tracer."""
    # Create config
    llm_config = LLMConfig(client=LLMClient.openai, model="gpt-4")
    
    # Create thread
    thread = ChatThread(name="Traced Thread", llm_config=llm_config)
    
    # Add messages
    thread.new_message = "First traced message"
    thread.add_user_message()
    
    thread.new_message = "Second traced message"
    thread.add_user_message()
    
    return thread


def test_entity_tracer_subentity_registration(setup_sql_storage):
    """Test that entity_tracer properly registers sub-entities."""
    
    # Create a thread with messages using a traced function
    thread = create_thread_with_messages()
    
    # Verify the thread was registered
    retrieved_thread = ChatThread.get(thread.ecs_id)
    assert retrieved_thread is not None
    assert retrieved_thread.name == "Traced Thread"
    
    # Verify that messages were registered
    assert len(retrieved_thread.history) == 2
    assert retrieved_thread.history[0].content == "First traced message"
    assert retrieved_thread.history[1].content == "Second traced message"


def test_entity_modification_with_subentities(setup_sql_storage):
    """Test that modifying sub-entities triggers proper forking and registration."""
    
    # Start with a thread with one message
    llm_config = LLMConfig(client=LLMClient.openai, model="gpt-4")
    thread = ChatThread(name="Modification Test", llm_config=llm_config)
    thread.new_message = "Initial message"
    thread.add_user_message()
    
    # Register the thread
    registered_thread = EntityRegistry.register(thread)
    
    # Verify initial state
    assert len(registered_thread.history) == 1
    assert registered_thread.history[0].content == "Initial message"
    
    # Modify the thread by adding a new message
    registered_thread.new_message = "Second message"
    registered_thread.add_user_message()
    
    # Register the modified thread
    updated_thread = EntityRegistry.register(registered_thread)
    
    # Verify the thread was forked (should have a new ID)
    assert updated_thread.ecs_id != registered_thread.ecs_id
    assert updated_thread.parent_id == registered_thread.ecs_id
    
    # Verify both messages are in the updated thread
    assert len(updated_thread.history) == 2
    assert updated_thread.history[0].content == "Initial message"
    assert updated_thread.history[1].content == "Second message"
    
    # Verify we can retrieve the original version
    original_thread = ChatThread.get(registered_thread.ecs_id)
    assert original_thread is not None
    assert len(original_thread.history) == 1
    assert original_thread.history[0].content == "Initial message"


if __name__ == "__main__":
    # Set up the SQL storage
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    from minference.threads.sql_models import (
        Base, ChatThreadSQL, ChatMessageSQL, LLMConfigSQL, 
        SystemPromptSQL, ToolSQL, CallableToolSQL, StructuredToolSQL,
        UsageSQL, GeneratedJsonObjectSQL, RawOutputSQL, ProcessedOutputSQL
    )
    from minference.ecs.storage import BaseEntitySQL
    
    engine = create_engine("sqlite:///:memory:", echo=True)
    
    # Explicitly create the base_entity table
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE base_entity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ecs_id CHAR(32) NOT NULL,
            lineage_id CHAR(32) NOT NULL,
            parent_id CHAR(32),
            created_at DATETIME NOT NULL,
            old_ids JSON NOT NULL,
            class_name VARCHAR(255) NOT NULL,
            data JSON,
            entity_type VARCHAR(50) NOT NULL
        )
        """))
        conn.commit()
    
    # Create all other tables
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    
    # Add EntityRegistry to __main__
    sys.modules['__main__'].__dict__['EntityRegistry'] = EntityRegistry
    
    # Create SQL storage
    sql_storage = SqlEntityStorage(
        session_factory=session_factory,
        entity_to_orm_map=ENTITY_MODEL_MAP
    )
    
    # Use SQL storage
    EntityRegistry.use_storage(sql_storage)
    
    # Run the tests
    test_complex_nested_entities(None)
    test_entity_tracer_subentity_registration(None)
    test_entity_modification_with_subentities(None)
    
    print("All tests passed when run directly!")