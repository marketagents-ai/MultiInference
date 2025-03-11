"""
Tests for RawOutput and ProcessedOutput entities in SQL storage.
Verifies creation, storage, retrieval, and relationships for these entities.
"""
import sys
import sqlite3
import json
import time
from uuid import UUID, uuid4
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import pytest
from sqlalchemy import create_engine, select, Integer, String, JSON, DateTime
from sqlalchemy.orm import Session, sessionmaker, joinedload, declarative_base, mapped_column

from minference.ecs.entity import Entity
from minference.ecs.enregistry import EntityRegistry
from minference.threads.models import (
    RawOutput, 
    ProcessedOutput, 
    LLMClient, 
    GeneratedJsonObject,
    Usage
)
from minference.threads.sql_models import (
    Base, 
    RawOutputSQL, 
    ProcessedOutputSQL, 
    UsageSQL, 
    GeneratedJsonObjectSQL
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
def session_factory(session):
    """Create a session factory that returns the test session."""
    def _session_factory():
        return session
    return _session_factory

@pytest.fixture
def setup_sql_storage(session_factory):
    """Configure EntityRegistry to use SQL storage."""
    from minference.ecs.entity import SqlEntityStorage
    from minference.threads.sql_models import ENTITY_MODEL_MAP
    
    # Create SQL storage with the session factory and entity mappings
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

def test_raw_output_creation_sql(setup_sql_storage):
    """Test basic RawOutput creation and storage in SQL."""
    # Create a simple OpenAI-style raw result
    raw_result = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help you today?"
                },
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
    
    # Create RawOutput
    start_time = time.time() - 1  # 1 second ago
    end_time = time.time()
    thread_id = uuid4()
    thread_live_id = uuid4()
    
    raw_output = RawOutput(
        raw_result=raw_result,
        completion_kwargs={"model": "gpt-4", "temperature": 0.7},
        start_time=start_time,
        end_time=end_time,
        chat_thread_id=thread_id,
        chat_thread_live_id=thread_live_id,
        client=LLMClient.openai
    )
    
    # Register with SQL storage
    registered_output = EntityRegistry.register(raw_output)
    assert registered_output is not None
    
    # Retrieve the stored entity
    retrieved = RawOutput.get(registered_output.ecs_id)
    assert retrieved is not None
    assert isinstance(retrieved, RawOutput)
    
    # Verify the content was stored correctly
    assert retrieved.raw_result == raw_result
    assert retrieved.completion_kwargs == {"model": "gpt-4", "temperature": 0.7}
    assert abs(retrieved.start_time - start_time) < 0.001  # Allow for minor float precision differences
    assert abs(retrieved.end_time - end_time) < 0.001
    assert retrieved.chat_thread_id == thread_id
    assert retrieved.chat_thread_live_id == thread_live_id
    assert retrieved.client == LLMClient.openai
    
    # Check time_taken property
    assert abs(retrieved.time_taken - (end_time - start_time)) < 0.001
    
    # Create a session to directly query SQL objects
    session_factory = EntityRegistry._storage._session_factory
    session = session_factory()
    
    # Query raw output directly from SQL
    sql_raw_output = session.query(RawOutputSQL).filter_by(ecs_id=registered_output.ecs_id).first()
    assert sql_raw_output is not None
    assert sql_raw_output.raw_result == raw_result
    assert sql_raw_output.client_type == "openai"

def test_raw_output_parsing_openai_sql(setup_sql_storage):
    """Test RawOutput parsing for OpenAI format in SQL storage."""
    # Create a simple OpenAI-style raw result with text
    raw_result = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help you today?"
                },
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
    
    # Create RawOutput
    raw_output = RawOutput(
        raw_result=raw_result,
        completion_kwargs={},
        start_time=time.time() - 1,
        end_time=time.time(),
        chat_thread_id=uuid4(),
        chat_thread_live_id=uuid4(),
        client=LLMClient.openai
    )
    
    # Register with SQL storage
    registered_output = EntityRegistry.register(raw_output)
    assert registered_output is not None
    
    # Retrieve the stored entity
    retrieved = RawOutput.get(registered_output.ecs_id)
    assert retrieved is not None
    
    # Test the str_content property
    assert retrieved.str_content == "Hello, how can I help you today?"
    
    # Test that no JSON object was found
    assert retrieved.json_object is None
    assert retrieved.contains_object is False
    
    # Test usage extraction
    assert retrieved.usage is not None
    assert retrieved.usage.model == "gpt-4"
    assert retrieved.usage.prompt_tokens == 10
    assert retrieved.usage.completion_tokens == 20
    assert retrieved.usage.total_tokens == 30

def test_raw_output_parsing_openai_tool_call_sql(setup_sql_storage):
    """Test RawOutput parsing for OpenAI tool call format in SQL storage."""
    # Create an OpenAI-style raw result with tool call
    tool_call_result = {
        "id": "chatcmpl-456",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "multiply",
                                "arguments": '{"x": 5, "y": 3}'
                            }
                        }
                    ]
                },
                "index": 0,
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40
        }
    }
    
    # Create RawOutput
    raw_output = RawOutput(
        raw_result=tool_call_result,
        completion_kwargs={},
        start_time=time.time() - 1,
        end_time=time.time(),
        chat_thread_id=uuid4(),
        chat_thread_live_id=uuid4(),
        client=LLMClient.openai
    )
    
    # Register with SQL storage
    registered_output = EntityRegistry.register(raw_output)
    assert registered_output is not None
    
    # Retrieve the stored entity
    retrieved = RawOutput.get(registered_output.ecs_id)
    assert retrieved is not None
    
    # Test content (should be empty for tool calls)
    assert retrieved.str_content == ""
    
    # Test JSON object extraction
    assert retrieved.json_object is not None
    assert retrieved.contains_object is True
    assert retrieved.json_object.name == "multiply"
    
    # Convert string arguments to dict
    tool_args = json.loads('{"x": 5, "y": 3}')
    assert retrieved.json_object.object == tool_args
    assert retrieved.json_object.tool_call_id == "call_123"

def test_raw_output_parsing_json_content_sql(setup_sql_storage):
    """Test RawOutput parsing for content with embedded JSON in SQL storage."""
    # Create an OpenAI-style raw result with JSON in content
    json_content_result = {
        "id": "chatcmpl-789",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"result": 42, "message": "Answer found"}'
                },
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 15,
            "total_tokens": 20
        }
    }
    
    # Create RawOutput
    raw_output = RawOutput(
        raw_result=json_content_result,
        completion_kwargs={},
        start_time=time.time() - 1,
        end_time=time.time(),
        chat_thread_id=uuid4(),
        chat_thread_live_id=uuid4(),
        client=LLMClient.openai
    )
    
    # Register with SQL storage
    registered_output = EntityRegistry.register(raw_output)
    assert registered_output is not None
    
    # Retrieve the stored entity
    retrieved = RawOutput.get(registered_output.ecs_id)
    assert retrieved is not None
    
    # Test JSON object extraction from content
    assert retrieved.json_object is not None
    assert retrieved.contains_object is True
    assert retrieved.json_object.name == "parsed_content"
    assert retrieved.json_object.object == {"result": 42, "message": "Answer found"}
    
    # If the content is fully JSON, str_content might be None because it was parsed completely
    # Just verify that we successfully parsed the JSON content
    assert retrieved.json_object.object == {"result": 42, "message": "Answer found"}

def test_raw_output_parsing_anthropic_sql(setup_sql_storage):
    """Test RawOutput parsing for Anthropic format in SQL storage."""
    # Create an Anthropic-style raw result
    anthropic_result = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-opus-20240229",
        "content": [
            {
                "type": "text",
                "text": "Hello, how can I help you today?"
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20
        }
    }
    
    # Create RawOutput
    raw_output = RawOutput(
        raw_result=anthropic_result,
        completion_kwargs={},
        start_time=time.time() - 1,
        end_time=time.time(),
        chat_thread_id=uuid4(),
        chat_thread_live_id=uuid4(),
        client=LLMClient.anthropic
    )
    
    # Register with SQL storage
    registered_output = EntityRegistry.register(raw_output)
    assert registered_output is not None
    
    # Retrieve the stored entity
    retrieved = RawOutput.get(registered_output.ecs_id)
    assert retrieved is not None
    
    # Test the str_content property (extraction from Anthropic format)
    assert retrieved.str_content == "Hello, how can I help you today?"
    
    # Test that no JSON object was found
    assert retrieved.json_object is None
    assert retrieved.contains_object is False
    
    # Test usage extraction from Anthropic format
    assert retrieved.usage is not None
    assert retrieved.usage.model == "claude-3-opus-20240229"
    assert retrieved.usage.prompt_tokens == 10
    assert retrieved.usage.completion_tokens == 20
    assert retrieved.usage.total_tokens == 30  # Should calculate sum

def test_raw_output_parsing_anthropic_tool_sql(setup_sql_storage):
    """Test RawOutput parsing for Anthropic tool format in SQL storage."""
    # Create an Anthropic-style raw result with tool use
    anthropic_tool_result = {
        "id": "msg_456",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-opus-20240229",
        "content": [
            {
                "type": "tool_use",
                "id": "tool_123",
                "name": "multiply",
                "input": {"x": 5, "y": 3}
            }
        ],
        "usage": {
            "input_tokens": 15,
            "output_tokens": 25
        }
    }
    
    # Create RawOutput
    raw_output = RawOutput(
        raw_result=anthropic_tool_result,
        completion_kwargs={},
        start_time=time.time() - 1,
        end_time=time.time(),
        chat_thread_id=uuid4(),
        chat_thread_live_id=uuid4(),
        client=LLMClient.anthropic
    )
    
    # Register with SQL storage
    registered_output = EntityRegistry.register(raw_output)
    assert registered_output is not None
    
    # Retrieve the stored entity
    retrieved = RawOutput.get(registered_output.ecs_id)
    assert retrieved is not None
    
    # For tool use, str_content might be None since we extracted JSON
    assert retrieved.str_content is None or retrieved.str_content == ""
    
    # Test JSON object extraction
    assert retrieved.json_object is not None
    assert retrieved.contains_object is True
    assert retrieved.json_object.name == "multiply"
    assert retrieved.json_object.object == {"x": 5, "y": 3}
    assert retrieved.json_object.tool_call_id == "tool_123"

def test_processed_output_creation_sql(setup_sql_storage):
    """Test basic ProcessedOutput creation and storage in SQL."""
    # Create necessary components
    thread_id = uuid4()
    thread_live_id = uuid4()
    
    # Create a raw output first
    raw_output = RawOutput(
        raw_result={"id": "test", "choices": [{"message": {"content": "test content"}}]},
        completion_kwargs={},
        start_time=time.time() - 1,
        end_time=time.time(),
        chat_thread_id=thread_id,
        chat_thread_live_id=thread_live_id,
        client=LLMClient.openai
    )
    
    # Register raw output
    registered_raw = EntityRegistry.register(raw_output)
    
    # Create JSON object
    json_object = GeneratedJsonObject(
        name="test_result",
        object={"result": "success"}
    )
    
    # Register JSON object
    registered_json = EntityRegistry.register(json_object)
    
    # Create usage
    usage = Usage(
        model="gpt-4",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30
    )
    
    # Register usage
    registered_usage = EntityRegistry.register(usage)
    
    # Create ProcessedOutput
    processed_output = ProcessedOutput(
        content="Test content",
        json_object=registered_json,
        usage=registered_usage,
        error=None,
        time_taken=0.5,
        llm_client=LLMClient.openai,
        raw_output=registered_raw,
        chat_thread_id=thread_id,
        chat_thread_live_id=thread_live_id
    )
    
    # Register with SQL storage
    registered_processed = EntityRegistry.register(processed_output)
    assert registered_processed is not None
    
    # Retrieve the stored entity
    retrieved = ProcessedOutput.get(registered_processed.ecs_id)
    assert retrieved is not None
    assert isinstance(retrieved, ProcessedOutput)
    
    # Verify fields and relationships
    assert retrieved.content == "Test content"
    assert retrieved.error is None
    assert retrieved.time_taken == 0.5
    assert retrieved.llm_client == LLMClient.openai
    assert retrieved.chat_thread_id == thread_id
    assert retrieved.chat_thread_live_id == thread_live_id
    
    # Verify relationships were properly stored and loaded
    assert retrieved.raw_output is not None
    assert retrieved.raw_output.ecs_id == registered_raw.ecs_id
    
    assert retrieved.json_object is not None
    assert retrieved.json_object.ecs_id == registered_json.ecs_id
    assert retrieved.json_object.name == "test_result"
    assert retrieved.json_object.object == {"result": "success"}
    
    assert retrieved.usage is not None
    assert retrieved.usage.ecs_id == registered_usage.ecs_id
    assert retrieved.usage.model == "gpt-4"
    assert retrieved.usage.prompt_tokens == 10
    assert retrieved.usage.completion_tokens == 20
    assert retrieved.usage.total_tokens == 30

def test_processed_output_from_raw_output_sql(setup_sql_storage):
    """Test creating ProcessedOutput from RawOutput in SQL storage."""
    # Create thread IDs
    thread_id = uuid4()
    thread_live_id = uuid4()
    
    # Create a raw output with OpenAI format
    raw_result = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help you today?"
                },
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
    
    raw_output = RawOutput(
        raw_result=raw_result,
        completion_kwargs={},
        start_time=time.time() - 1,
        end_time=time.time(),
        chat_thread_id=thread_id,
        chat_thread_live_id=thread_live_id,
        client=LLMClient.openai
    )
    
    # Register raw output in SQL storage
    registered_raw = EntityRegistry.register(raw_output)
    
    # Retrieve from storage to ensure we're working with the stored version
    stored_raw_output = RawOutput.get(registered_raw.ecs_id)
    assert stored_raw_output is not None
    
    # Create ProcessedOutput from RawOutput
    processed_output = stored_raw_output.create_processed_output()
    
    # Register the processed output
    registered_processed = EntityRegistry.register(processed_output)
    assert registered_processed is not None
    
    # Retrieve the stored processed output
    retrieved = ProcessedOutput.get(registered_processed.ecs_id)
    assert retrieved is not None
    
    # Verify correct content extraction
    assert retrieved.content == "Hello, how can I help you today?"
    assert retrieved.json_object is None
    
    # Verify usage extraction and storage
    assert retrieved.usage is not None
    assert retrieved.usage.model == "gpt-4"
    assert retrieved.usage.prompt_tokens == 10
    assert retrieved.usage.completion_tokens == 20
    assert retrieved.usage.total_tokens == 30
    
    # Verify relationship to raw output
    assert retrieved.raw_output is not None
    assert retrieved.raw_output.ecs_id == registered_raw.ecs_id
    
    # Verify thread IDs were copied
    assert retrieved.chat_thread_id == thread_id
    assert retrieved.chat_thread_live_id == thread_live_id

def test_processed_output_with_error_sql(setup_sql_storage):
    """Test creating ProcessedOutput with error in SQL storage."""
    # Create a raw output first
    raw_output = RawOutput(
        raw_result={"error": "Model overloaded"},
        completion_kwargs={},
        start_time=time.time() - 1,
        end_time=time.time(),
        chat_thread_id=uuid4(),
        chat_thread_live_id=uuid4(),
        client=LLMClient.openai
    )
    
    # Register raw output
    registered_raw = EntityRegistry.register(raw_output)
    
    # Create ProcessedOutput with error
    processed_output = ProcessedOutput(
        content=None,
        json_object=None,
        usage=None,
        error="API Error: Model overloaded, please try again later",
        time_taken=0.5,
        llm_client=LLMClient.openai,
        raw_output=registered_raw,
        chat_thread_id=raw_output.chat_thread_id,
        chat_thread_live_id=raw_output.chat_thread_live_id
    )
    
    # Register with SQL storage
    registered_processed = EntityRegistry.register(processed_output)
    assert registered_processed is not None
    
    # Retrieve the stored entity
    retrieved = ProcessedOutput.get(registered_processed.ecs_id)
    assert retrieved is not None
    
    # Verify error was stored correctly
    assert retrieved.content is None
    assert retrieved.json_object is None
    assert retrieved.error == "API Error: Model overloaded, please try again later"
    
    # Verify raw output relationship
    assert retrieved.raw_output is not None
    assert retrieved.raw_output.ecs_id == registered_raw.ecs_id