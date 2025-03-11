"""
Tests for entity tracing behavior with SQL storage for ChatThread operations.

This module focuses on testing:
1. Tracing behavior when adding user messages to a ChatThread with SQL storage
2. Tracing behavior during chat turn additions
3. Verification of correct entity versioning during threaded conversations
4. SQL storage persistence of versioned entities
"""

import sys
import sqlite3
import json
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Set, cast, Union, Any
from uuid import UUID

import pytest
from sqlalchemy import create_engine, select, Integer, String, JSON, DateTime
from sqlalchemy.orm import Session, sessionmaker, joinedload

from minference.ecs.entity import Entity, EntityRegistry, entity_tracer, Base as EntityBase_Base
from minference.threads.models import (
    ChatThread, ChatMessage, LLMConfig, LLMClient, MessageRole, SystemPrompt,
    CallableTool, StructuredTool, GeneratedJsonObject, Usage, ResponseFormat,
    RawOutput, ProcessedOutput
)
from minference.threads.sql_models import (
    Base as ThreadBase, ChatMessageSQL, ChatThreadSQL
)

# Add EntityRegistry to __main__ for entity methods - this is REQUIRED 
sys.modules['__main__'].__dict__['EntityRegistry'] = EntityRegistry

# Setup SQLite UUID handling
def adapt_uuid(uuid):
    return str(uuid)

def convert_uuid(s):
    return UUID(s.decode())

sqlite3.register_adapter(UUID, adapt_uuid)
sqlite3.register_converter("uuid", convert_uuid)

@pytest.fixture
def engine():
    """Create an in-memory SQLite engine."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    
    # Import the Base from entity.py and sql_models.py to create all tables
    from minference.ecs.entity import BaseEntitySQL, Base as EntityBase_Base
    from minference.threads.sql_models import Base as ThreadBase
    
    # Create all tables explicitly to ensure they exist
    ThreadBase.metadata.create_all(engine)
    EntityBase_Base.metadata.create_all(engine)
    
    return engine

@pytest.fixture
def session(engine):
    """Create a database session."""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

@pytest.fixture
def setup_sql_storage(engine):
    """Set up SQL storage with EntityRegistry."""
    from minference.ecs.entity import SqlEntityStorage
    from minference.threads.sql_models import ENTITY_MODEL_MAP
    
    # Create SqlEntityStorage with entity mappings
    session_factory = sessionmaker(bind=engine)
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

def test_document_sub_entity_behavior(setup_sql_storage):
    """
    Document the behavior of sub-entity registration with SQL storage.
    
    This test shows how we have to handle sub-entities with the current implementation.
    """
    print("\n=== TESTING SUB-ENTITY REGISTRATION BEHAVIOR ===")
    
    # Create test thread
    thread = ChatThread(
        name="Test Sub-Entity Registration",
        llm_config=LLMConfig(
            model="gpt-4-turbo", 
            client=LLMClient.openai
        )
    )
    registered_thread = EntityRegistry.register(thread)
    thread_id = registered_thread.ecs_id
    print(f"Created and registered thread: {thread_id}")
    
    # Add a message to the thread
    registered_thread.new_message = "Test message for sub-entity documentation"
    message = registered_thread.add_user_message()
    message_id = message.ecs_id
    print(f"Added message to thread: {message_id}")
    
    # Verify dependency graph works correctly
    registered_thread.initialize_deps_graph()
    sub_entities = registered_thread.get_sub_entities()
    print(f"Sub-entities found in dependency graph: {len(sub_entities)}")
    
    # IMPORTANT: We have to explicitly register the message too!
    print("\n--- Explicitly registering the message ---")
    registered_message = EntityRegistry.register(message)
    print(f"Registered message: {registered_message.ecs_id}")
    
    # Now update the thread
    print("\n--- Registering the updated thread ---")
    updated_thread = EntityRegistry.register(registered_thread)
    print(f"Updated thread ID: {updated_thread.ecs_id}")
    
    # Check both can be retrieved
    retrieved_message = ChatMessage.get(message_id)
    print(f"\nMessage retrieval: {'FOUND' if retrieved_message else 'NOT FOUND'}")
    
    retrieved_thread = ChatThread.get(updated_thread.ecs_id)
    print(f"Thread retrieval: {'FOUND' if retrieved_thread else 'NOT FOUND'}")
    print(f"Thread history count: {len(retrieved_thread.history) if retrieved_thread else 'N/A'}")
    
    # Verify the key relationship is maintained
    if retrieved_thread and retrieved_thread.history:
        print(f"History message ID: {retrieved_thread.history[0].ecs_id}")
        print(f"History message matches original: {retrieved_thread.history[0].ecs_id == message_id}")
    
    print("\n=== CONCLUSION ===")
    print("When using SQL storage, sub-entities like messages must be explicitly registered")
    print("The entity_tracer does not correctly handle sub-entity registration automatically")
    print("This requires careful manual handling of every sub-entity in the application code")
    
    # These assertions confirm how things currently work
    assert retrieved_message is not None, "Message should be found after explicit registration"
    assert retrieved_thread is not None, "Thread should be retrievable"
    assert len(retrieved_thread.history) > 0, "Thread should include registered message in history"
    assert retrieved_thread.history[0].ecs_id == message_id, "Retrieved message should match original"
    
    # Verify lineage tracking
    lineage = EntityRegistry.get_lineage_entities(updated_thread.lineage_id)
    lineage_threads = [e for e in lineage if isinstance(e, ChatThread)]
    assert len(lineage_threads) == 2  # Original and updated thread
    
    # Verify the message is stored in EntityRegistry
    message = ChatMessage.get(updated_thread.history[0].ecs_id)
    assert message is not None
    assert message.content == "Hello, world!"
    assert message.chat_thread_id == updated_thread.ecs_id

def test_multiple_user_messages_tracing(setup_sql_storage):
    """Test tracing behavior when adding multiple user messages."""
    # Create a thread with config
    llm_config = LLMConfig(
        model="gpt-4-turbo",
        client=LLMClient.openai
    )
    registered_config = EntityRegistry.register(llm_config)
    
    thread = ChatThread(
        name="Multi-Message Thread",
        llm_config=registered_config
    )
    registered_thread = EntityRegistry.register(thread)
    original_id = registered_thread.ecs_id
    
    # Add first message
    registered_thread.new_message = "First message"
    msg1 = registered_thread.add_user_message()
    EntityRegistry.register(msg1)
    thread_v1 = EntityRegistry.register(registered_thread)
    
    # Get a fresh copy and add second message
    thread_fresh = ChatThread.get(thread_v1.ecs_id)
    thread_fresh.new_message = "Second message"
    msg2 = thread_fresh.add_user_message()
    EntityRegistry.register(msg2)
    thread_v2 = EntityRegistry.register(thread_fresh)
    
    # Get another fresh copy and add third message
    thread_fresher = ChatThread.get(thread_v2.ecs_id)
    thread_fresher.new_message = "Third message"
    msg3 = thread_fresher.add_user_message()
    EntityRegistry.register(msg3)
    thread_v3 = EntityRegistry.register(thread_fresher)
    
    # Verify we have distinct versions
    assert original_id != thread_v1.ecs_id
    assert thread_v1.ecs_id != thread_v2.ecs_id
    assert thread_v2.ecs_id != thread_v3.ecs_id
    
    # Verify correct history lengths
    original = ChatThread.get(original_id)
    assert len(original.history) == 0
    
    v1 = ChatThread.get(thread_v1.ecs_id)
    assert len(v1.history) == 1
    assert v1.history[0].content == "First message"
    
    v2 = ChatThread.get(thread_v2.ecs_id)
    assert len(v2.history) == 2
    assert v2.history[1].content == "Second message"
    
    v3 = ChatThread.get(thread_v3.ecs_id)
    assert len(v3.history) == 3
    assert v3.history[2].content == "Third message"
    
    # Verify all versions are in the lineage
    lineage = EntityRegistry.get_lineage_entities(thread_v3.lineage_id)
    thread_versions = [e for e in lineage if isinstance(e, ChatThread)]
    assert len(thread_versions) == 4  # Original + 3 versions
    
    # Verify correct parent-child relationships in messages
    assert v3.history[1].parent_message_uuid == v3.history[0].ecs_id
    assert v3.history[2].parent_message_uuid == v3.history[1].ecs_id

@pytest.mark.asyncio
async def test_add_chat_turn_tracing(setup_sql_storage):
    """Test tracing behavior when adding a complete chat turn."""
    # Create a thread with config
    llm_config = LLMConfig(
        model="gpt-4-turbo",
        client=LLMClient.openai,
        response_format=ResponseFormat.text
    )
    registered_config = EntityRegistry.register(llm_config)
    
    thread = ChatThread(
        name="Chat Turn Thread",
        llm_config=registered_config
    )
    registered_thread = EntityRegistry.register(thread)
    
    # Add a user message first
    registered_thread.new_message = "Hello assistant!"
    registered_thread.add_user_message()
    thread_with_user_msg = EntityRegistry.register(registered_thread)
    user_msg_id = thread_with_user_msg.ecs_id
    
    # Create a simulated LLM response
    raw_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": "gpt-4-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    }
    
    # Create RawOutput and ProcessedOutput with all required fields
    current_time = time.time()
    raw_output = RawOutput(
        raw_result=raw_response,
        completion_kwargs={},
        start_time=current_time - 1.0,
        end_time=current_time,
        client=LLMClient.openai,
        chat_thread_id=thread_with_user_msg.ecs_id,
        chat_thread_live_id=thread_with_user_msg.live_id
    )
    registered_raw = EntityRegistry.register(raw_output)
    
    # Process the output
    processed_output = ProcessedOutput.from_raw_output(registered_raw)
    registered_processed = EntityRegistry.register(processed_output)
    
    # Get a fresh thread copy and add the chat turn
    thread_fresh = ChatThread.get(user_msg_id)
    await thread_fresh.add_chat_turn_history(registered_processed)
    thread_with_response = EntityRegistry.register(thread_fresh)
    
    # Verify a new version was created
    assert thread_with_response.ecs_id != user_msg_id
    
    # Verify the thread contains both messages
    assert len(thread_with_response.history) == 2
    assert thread_with_response.history[0].role == MessageRole.user
    assert thread_with_response.history[1].role == MessageRole.assistant
    assert thread_with_response.history[1].content == "Hello! How can I help you today?"
    
    # Verify the messages are linked correctly
    assert thread_with_response.history[1].parent_message_uuid == thread_with_response.history[0].ecs_id
    
    # Verify usage was associated with the assistant message
    assert thread_with_response.history[1].usage is not None
    assert thread_with_response.history[1].usage.prompt_tokens == 10
    assert thread_with_response.history[1].usage.completion_tokens == 8
    
    # Verify all versions can be retrieved from storage
    original = ChatThread.get(registered_thread.ecs_id)
    assert len(original.history) == 0
    
    user_msg_thread = ChatThread.get(user_msg_id)
    assert len(user_msg_thread.history) == 1
    
    response_thread = ChatThread.get(thread_with_response.ecs_id)
    assert len(response_thread.history) == 2
    
    # Verify all messages can be retrieved individually
    user_message = ChatMessage.get(thread_with_response.history[0].ecs_id)
    assert user_message is not None
    
    assistant_message = ChatMessage.get(thread_with_response.history[1].ecs_id)
    assert assistant_message is not None
    
    # Verify lineage contains all versions
    lineage = EntityRegistry.get_lineage_entities(thread_with_response.lineage_id)
    thread_versions = [e for e in lineage if isinstance(e, ChatThread)]
    assert len(thread_versions) == 3  # Original + user msg + response

@pytest.mark.asyncio
async def test_tool_execution_tracing(setup_sql_storage):
    """Test tracing behavior when executing tools in a chat turn."""
    # Define a simple tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b
    
    # Register with CallableRegistry first
    from minference.ecs.caregistry import CallableRegistry
    CallableRegistry.register("add", add_numbers)
    
    # Create a callable tool
    tool = CallableTool(
        name="add",
        docstring="Add two numbers",
        input_schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
        output_schema={"type": "integer"}
    )
    registered_tool = EntityRegistry.register(tool)
    
    # Create a thread with the tool
    llm_config = LLMConfig(
        model="gpt-4-turbo",
        client=LLMClient.openai,
        response_format=ResponseFormat.text
    )
    registered_config = EntityRegistry.register(llm_config)
    
    thread = ChatThread(
        name="Tool Execution Thread",
        llm_config=registered_config,
        tools=[registered_tool]
    )
    registered_thread = EntityRegistry.register(thread)
    
    # Add a user message
    registered_thread.new_message = "Please add 5 and 7"
    msg = registered_thread.add_user_message()
    EntityRegistry.register(msg)
    thread_with_user_msg = EntityRegistry.register(registered_thread)
    user_msg_id = thread_with_user_msg.ecs_id
    
    # Create a simulated LLM response with tool call
    raw_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": "gpt-4-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 5, "b": 7}'
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 10,
            "total_tokens": 25
        }
    }
    
    # Create and process the output
    import time
    current_time = time.time()
    raw_output = RawOutput(
        raw_result=raw_response,
        completion_kwargs={},
        start_time=current_time - 1.0,
        end_time=current_time,
        client=LLMClient.openai,
        chat_thread_id=thread_with_user_msg.ecs_id,
        chat_thread_live_id=thread_with_user_msg.live_id
    )
    registered_raw = EntityRegistry.register(raw_output)
    
    processed_output = registered_raw.create_processed_output()
    registered_processed = EntityRegistry.register(processed_output)
    
    # Get a fresh thread copy and add the chat turn
    thread_fresh = ChatThread.get(user_msg_id)
    await thread_fresh.add_chat_turn_history(registered_processed)
    thread_with_tool_call = EntityRegistry.register(thread_fresh)
    
    # Verify the thread now contains 3 messages (user, assistant, tool)
    assert len(thread_with_tool_call.history) == 3
    assert thread_with_tool_call.history[0].role == MessageRole.user
    assert thread_with_tool_call.history[1].role == MessageRole.assistant
    assert thread_with_tool_call.history[2].role == MessageRole.tool
    
    # Verify the tool message contains the result
    tool_result = json.loads(thread_with_tool_call.history[2].content)
    assert tool_result == 12
    
    # Verify all three versions of the thread are in storage
    original = ChatThread.get(registered_thread.ecs_id)
    assert len(original.history) == 0
    
    user_msg_thread = ChatThread.get(user_msg_id)
    assert len(user_msg_thread.history) == 1
    
    tool_thread = ChatThread.get(thread_with_tool_call.ecs_id)
    assert len(tool_thread.history) == 3
    
    # Verify all entity IDs are different (entity_tracer is creating new versions)
    assert registered_thread.ecs_id != thread_with_user_msg.ecs_id
    assert thread_with_user_msg.ecs_id != thread_with_tool_call.ecs_id
    
    # Verify the EntityRegistry contains all the messages
    for msg in thread_with_tool_call.history:
        stored_msg = ChatMessage.get(msg.ecs_id)
        assert stored_msg is not None

@pytest.mark.asyncio
async def test_multiple_chat_turns_tracing(setup_sql_storage):
    """Test tracing with multiple complete chat turns."""
    # Create a thread with config
    llm_config = LLMConfig(
        model="gpt-4-turbo",
        client=LLMClient.openai,
        response_format=ResponseFormat.text
    )
    registered_config = EntityRegistry.register(llm_config)
    
    thread = ChatThread(
        name="Multiple Chat Turns Thread",
        llm_config=registered_config
    )
    registered_thread = EntityRegistry.register(thread)
    original_id = registered_thread.ecs_id
    
    # First turn: User message
    registered_thread.new_message = "Hello!"
    msg1 = registered_thread.add_user_message()
    EntityRegistry.register(msg1)
    thread_v1 = EntityRegistry.register(registered_thread)
    
    # First turn: Assistant response
    raw_response1 = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": "gpt-4-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hi there! How can I help you today?"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 9,
            "total_tokens": 14
        }
    }
    
    current_time = time.time()
    raw_output1 = RawOutput(
        raw_result=raw_response1,
        completion_kwargs={},
        start_time=current_time - 1.0,
        end_time=current_time,
        client=LLMClient.openai,
        chat_thread_id=thread_v1.ecs_id,
        chat_thread_live_id=thread_v1.live_id
    )
    registered_raw1 = EntityRegistry.register(raw_output1)
    
    processed_output1 = registered_raw1.create_processed_output()
    registered_processed1 = EntityRegistry.register(processed_output1)
    
    thread_fresh1 = ChatThread.get(thread_v1.ecs_id)
    await thread_fresh1.add_chat_turn_history(registered_processed1)
    thread_v2 = EntityRegistry.register(thread_fresh1)
    
    # Second turn: User message
    thread_fresh2 = ChatThread.get(thread_v2.ecs_id)
    thread_fresh2.new_message = "What's the weather today?"
    msg2 = thread_fresh2.add_user_message()
    EntityRegistry.register(msg2)
    thread_v3 = EntityRegistry.register(thread_fresh2)
    
    # Second turn: Assistant response
    raw_response2 = {
        "id": "chatcmpl-456",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": "gpt-4-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I don't have real-time weather data. You would need to check a weather service for current conditions."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 20,
            "total_tokens": 45
        }
    }
    
    current_time = time.time()
    raw_output2 = RawOutput(
        raw_result=raw_response2,
        completion_kwargs={},
        start_time=current_time - 1.0,
        end_time=current_time,
        client=LLMClient.openai,
        chat_thread_id=thread_v3.ecs_id,
        chat_thread_live_id=thread_v3.live_id
    )
    registered_raw2 = EntityRegistry.register(raw_output2)
    
    processed_output2 = registered_raw2.create_processed_output()
    registered_processed2 = EntityRegistry.register(processed_output2)
    
    thread_fresh3 = ChatThread.get(thread_v3.ecs_id)
    await thread_fresh3.add_chat_turn_history(registered_processed2)
    thread_v4 = EntityRegistry.register(thread_fresh3)
    
    # Verify all thread versions are different
    version_ids = {original_id, thread_v1.ecs_id, thread_v2.ecs_id, thread_v3.ecs_id, thread_v4.ecs_id}
    assert len(version_ids) == 5  # All IDs should be unique
    
    # Verify correct history lengths for each version
    original = ChatThread.get(original_id)
    assert len(original.history) == 0
    
    v1 = ChatThread.get(thread_v1.ecs_id)
    assert len(v1.history) == 1
    
    v2 = ChatThread.get(thread_v2.ecs_id)
    assert len(v2.history) == 2
    
    v3 = ChatThread.get(thread_v3.ecs_id)
    assert len(v3.history) == 3
    
    v4 = ChatThread.get(thread_v4.ecs_id)
    assert len(v4.history) == 4
    
    # Verify correct message sequence and parent-child relationships
    assert v4.history[0].role == MessageRole.user
    assert v4.history[0].content == "Hello!"
    
    assert v4.history[1].role == MessageRole.assistant
    assert "Hi there!" in v4.history[1].content
    assert v4.history[1].parent_message_uuid == v4.history[0].ecs_id
    
    assert v4.history[2].role == MessageRole.user
    assert v4.history[2].content == "What's the weather today?"
    assert v4.history[2].parent_message_uuid == v4.history[1].ecs_id
    
    assert v4.history[3].role == MessageRole.assistant
    assert "weather" in v4.history[3].content
    assert v4.history[3].parent_message_uuid == v4.history[2].ecs_id
    
    # Verify correct usage tracking
    assert v4.history[1].usage is not None
    assert v4.history[1].usage.prompt_tokens == 5
    assert v4.history[1].usage.completion_tokens == 9
    
    assert v4.history[3].usage is not None
    assert v4.history[3].usage.prompt_tokens == 25
    assert v4.history[3].usage.completion_tokens == 20
    
    # Verify get_all_usages returns all assistant message usages
    all_usages = v4.get_all_usages()
    assert len(all_usages) == 2
    
    # Verify lineage contains all thread versions
    lineage = EntityRegistry.get_lineage_entities(v4.lineage_id)
    thread_versions = [e for e in lineage if isinstance(e, ChatThread)]
    assert len(thread_versions) == 5  # Original + 4 version updates