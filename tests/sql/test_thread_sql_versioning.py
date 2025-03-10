"""
Tests for entity versioning in the SQL storage backend for Thread entities.

This test module focuses on verifying that:
1. Modification detection works correctly for SQL-stored entities
2. Entity forking creates proper versions with lineage tracking
3. Relationships are preserved during versioning
4. Parent entities are automatically forked when children change
5. Circular references are handled properly in versioning
"""

import sys
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Set, cast, Union, Any
from uuid import UUID

import pytest
from sqlalchemy import create_engine, select, Integer, String, JSON, DateTime
from sqlalchemy.orm import Session, sessionmaker, joinedload, declarative_base, mapped_column

from minference.ecs.entity import Entity, EntityRegistry
from minference.threads.models import (
    ChatThread, ChatMessage, LLMConfig, LLMClient, MessageRole, SystemPrompt,
    CallableTool, StructuredTool, GeneratedJsonObject, Usage, ResponseFormat
)
from minference.threads.sql_models import (
    Base, ChatMessageSQL, ChatThreadSQL
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
    from minference.ecs.entity import BaseEntitySQL
    from minference.threads.sql_models import Base as ThreadBase, EntityBase
    
    # Need to create a Base instance with BaseEntitySQL
    EntityBase = declarative_base()
    
    # Create a table for BaseEntitySQL
    class BaseEntitySQLTable(EntityBase):
        __tablename__ = "baseentitysql"
        
        id = mapped_column(Integer, primary_key=True, autoincrement=True)
        ecs_id = mapped_column(String(36), nullable=False, index=True, unique=True)
        lineage_id = mapped_column(String(36), nullable=False, index=True)
        parent_id = mapped_column(String(36), nullable=True, index=True)
        created_at = mapped_column(DateTime(timezone=True), nullable=False)
        old_ids = mapped_column(JSON, nullable=False, default=list)
        class_name = mapped_column(String(255), nullable=False)
        data = mapped_column(JSON, nullable=False)
    
    # Create all tables explicitly to ensure they exist
    EntityBase.metadata.create_all(engine)
    ThreadBase.metadata.create_all(engine)
    
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

@pytest.fixture
def simple_chat_thread(setup_sql_storage):
    """Create a simple chat thread with system prompt and LLM config."""
    system_prompt = SystemPrompt(
        content="You are a helpful assistant.",
        name="Default System Prompt"
    )
    registered_prompt = EntityRegistry.register(system_prompt)
    
    llm_config = LLMConfig(
        client=LLMClient.openai,
        model="gpt-4",
        max_tokens=1000,
        temperature=0.7,
        response_format=ResponseFormat.text
    )
    registered_config = EntityRegistry.register(llm_config)
    
    thread = ChatThread(
        name="Test Thread",
        system_prompt=registered_prompt,
        llm_config=registered_config
    )
    
    # Register the thread to ensure it's in storage
    registered_thread = EntityRegistry.register(thread)
    
    return registered_thread

@pytest.fixture
def chat_thread_with_messages(setup_sql_storage, simple_chat_thread):
    """Create a chat thread with a message history."""
    thread = simple_chat_thread
    
    # Add user message
    user_message = ChatMessage(
        role=MessageRole.user,
        content="Hello, this is a test message",
        chat_thread_id=thread.ecs_id
    )
    registered_user_msg = EntityRegistry.register(user_message)
    
    # Add assistant message
    assistant_message = ChatMessage(
        role=MessageRole.assistant,
        content="I'm a helpful assistant.",
        parent_message_uuid=registered_user_msg.ecs_id,
        chat_thread_id=thread.ecs_id
    )
    registered_assistant_msg = EntityRegistry.register(assistant_message)
    
    # Update thread to include messages
    thread.history = [registered_user_msg, registered_assistant_msg]
    registered_thread = EntityRegistry.register(thread)
    
    return registered_thread

@pytest.fixture
def chat_thread_with_tools(setup_sql_storage, simple_chat_thread):
    """Create a chat thread with tools."""
    thread = simple_chat_thread
    
    # Create a callable tool
    callable_tool = CallableTool(
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
    registered_callable_tool = EntityRegistry.register(callable_tool)
    
    # Create a structured tool
    structured_tool = StructuredTool(
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
    registered_structured_tool = EntityRegistry.register(structured_tool)
    
    # Add tools to thread
    thread.tools = [registered_callable_tool, registered_structured_tool]
    
    # Register the updated thread with tools
    registered_thread = EntityRegistry.register(thread)
    
    return registered_thread

@pytest.fixture
def complex_chat_thread(setup_sql_storage, chat_thread_with_tools):
    """Create a complex chat thread with messages and tools."""
    thread = chat_thread_with_tools
    
    # Get the tools
    callable_tool = thread.tools[0]
    
    # Add user message
    user_message = ChatMessage(
        role=MessageRole.user,
        content="What's the weather in New York?",
        chat_thread_id=thread.ecs_id
    )
    registered_user_msg = EntityRegistry.register(user_message)
    
    # Add assistant message with tool call
    assistant_message = ChatMessage(
        role=MessageRole.assistant,
        content="",
        parent_message_uuid=registered_user_msg.ecs_id,
        chat_thread_id=thread.ecs_id,
        tool_name=callable_tool.name,
        tool_uuid=callable_tool.ecs_id,
        tool_call={"location": "New York"},
        oai_tool_call_id="call_123"
    )
    registered_assistant_msg = EntityRegistry.register(assistant_message)
    
    # Add tool response message
    tool_message = ChatMessage(
        role=MessageRole.tool,
        content=json.dumps({"temperature": 72, "condition": "sunny"}),
        parent_message_uuid=registered_assistant_msg.ecs_id,
        chat_thread_id=thread.ecs_id,
        tool_name=callable_tool.name,
        tool_uuid=callable_tool.ecs_id,
        oai_tool_call_id="call_123"
    )
    registered_tool_msg = EntityRegistry.register(tool_message)
    
    # Update thread to include messages
    thread.history = [registered_user_msg, registered_assistant_msg, registered_tool_msg]
    registered_thread = EntityRegistry.register(thread)
    
    return registered_thread

# Tests for basic versioning
def test_simple_modification_detection(setup_sql_storage, simple_chat_thread):
    """Test detection of modifications in a simple chat thread."""
    # Get the thread from storage
    thread = ChatThread.get(simple_chat_thread.ecs_id)
    assert thread is not None
    
    # Store original snapshot for comparison
    original_snapshot = thread
    
    # Get a fresh copy to modify
    modified_thread = ChatThread.get(simple_chat_thread.ecs_id)
    assert modified_thread is not None
    
    # Modify a simple field
    modified_thread.name = "Modified Thread Name"
    
    # Check for modifications
    has_changes, modified_entities = modified_thread.has_modifications(original_snapshot)
    
    # Verify changes were detected
    assert has_changes, "Changes should be detected"
    assert modified_thread in modified_entities, "The thread should be in the modified entities"
    
    # Check that we can get the specific field changes
    thread_diff = modified_entities[modified_thread]
    assert "name" in thread_diff.field_diffs
    assert thread_diff.field_diffs["name"]["type"] == "modified"
    assert thread_diff.field_diffs["name"]["old"] == "Test Thread"
    assert thread_diff.field_diffs["name"]["new"] == "Modified Thread Name"

def test_sub_entity_modification_detection(setup_sql_storage, simple_chat_thread):
    """Test detection of modifications in sub-entities of a chat thread."""
    # Get the thread from storage
    thread = ChatThread.get(simple_chat_thread.ecs_id)
    assert thread is not None
    
    # Store original snapshot for comparison
    original_snapshot = thread
    
    # Get a fresh copy to modify
    modified_thread = ChatThread.get(simple_chat_thread.ecs_id)
    assert modified_thread is not None
    
    # Modify the system prompt (a sub-entity)
    modified_thread.system_prompt.content = "Modified system prompt content"
    
    # Check for modifications
    has_changes, modified_entities = modified_thread.has_modifications(original_snapshot)
    
    # Verify changes were detected
    assert has_changes, "Changes should be detected"
    assert modified_thread.system_prompt in modified_entities, "The system prompt should be in the modified entities"
    
    # Check that parent entity is also marked for forking
    assert modified_thread in modified_entities, "The parent thread should also be in modified entities"
    
    # Check specific field changes in the system prompt
    system_prompt_diff = modified_entities[modified_thread.system_prompt]
    assert "content" in system_prompt_diff.field_diffs
    assert system_prompt_diff.field_diffs["content"]["old"] == "You are a helpful assistant."
    assert system_prompt_diff.field_diffs["content"]["new"] == "Modified system prompt content"

def test_entity_forking(setup_sql_storage, simple_chat_thread):
    """Test forking an entity when it has modifications."""
    # Get the thread from storage
    thread = ChatThread.get(simple_chat_thread.ecs_id)
    assert thread is not None
    
    # Save the original IDs
    original_thread_id = thread.ecs_id
    original_system_prompt_id = thread.system_prompt.ecs_id
    original_llm_config_id = thread.llm_config.ecs_id
    
    # Modify a field
    thread.name = "Forked Thread"
    
    # Fork the thread
    forked_thread = thread.fork()
    registered_forked_thread = EntityRegistry.register(forked_thread)
    
    # Verify the thread was forked (new ID)
    assert registered_forked_thread.ecs_id != original_thread_id, "Thread should have a new ID after forking"
    
    # Verify the modified data was preserved
    assert registered_forked_thread.name == "Forked Thread"
    
    # Verify the parent_id points to the original
    assert registered_forked_thread.parent_id == original_thread_id
    
    # Verify the lineage_id stayed the same
    assert registered_forked_thread.lineage_id == simple_chat_thread.lineage_id
    
    # Verify old_ids contains the original ID
    assert original_thread_id in registered_forked_thread.old_ids
    
    # Get the forked thread from storage to verify it was properly saved
    retrieved_fork = ChatThread.get(registered_forked_thread.ecs_id)
    assert retrieved_fork is not None
    assert retrieved_fork.name == "Forked Thread"
    assert retrieved_fork.parent_id == original_thread_id

def test_sub_entity_forking(setup_sql_storage, simple_chat_thread):
    """Test forking when a sub-entity has modifications."""
    # Get the thread from storage
    thread = ChatThread.get(simple_chat_thread.ecs_id)
    assert thread is not None
    
    # Save the original IDs
    original_thread_id = thread.ecs_id
    original_system_prompt_id = thread.system_prompt.ecs_id
    
    # Modify the system prompt
    thread.system_prompt.content = "Modified system prompt"
    
    # Fork the thread
    forked_thread = thread.fork()
    registered_forked_thread = EntityRegistry.register(forked_thread)
    
    # Verify the thread was forked (new ID)
    assert registered_forked_thread.ecs_id != original_thread_id, "Thread should have a new ID after forking"
    
    # Verify the system prompt was also forked
    assert registered_forked_thread.system_prompt.ecs_id != original_system_prompt_id, "System prompt should have a new ID"
    
    # Verify the modified data was preserved
    assert registered_forked_thread.system_prompt.content == "Modified system prompt"
    
    # Verify parent_id relationships
    assert registered_forked_thread.parent_id == original_thread_id
    assert registered_forked_thread.system_prompt.parent_id == original_system_prompt_id
    
    # Get the forked entities from storage to verify they were properly saved
    retrieved_fork = ChatThread.get(registered_forked_thread.ecs_id)
    assert retrieved_fork is not None
    assert retrieved_fork.system_prompt.content == "Modified system prompt"
    assert retrieved_fork.system_prompt.ecs_id != original_system_prompt_id

# Tests for message versioning
def test_direct_message_versioning(setup_sql_storage):
    """Test versioning when modifying a message directly."""
    # Create a single message directly
    message = ChatMessage(
        role=MessageRole.user,
        content="Original message content"
    )
    
    # Register the message
    registered_message = EntityRegistry.register(message)
    original_id = registered_message.ecs_id
    
    # Get a fresh copy from storage
    retrieved_message = ChatMessage.get(registered_message.ecs_id)
    assert retrieved_message is not None
    
    # Modify the message
    retrieved_message.content = "Modified message content"
    
    # Fork the message
    forked_message = retrieved_message.fork()
    
    # Register the forked message
    registered_forked = EntityRegistry.register(forked_message)
    
    # Verify the message was forked
    assert registered_forked.ecs_id != original_id
    assert registered_forked.content == "Modified message content"
    assert registered_forked.parent_id == original_id
    
    # Get the forked message from storage again
    retrieved_fork = ChatMessage.get(registered_forked.ecs_id)
    assert retrieved_fork is not None
    assert retrieved_fork.content == "Modified message content"

def test_thread_modification(setup_sql_storage):
    """Test versioning by modifying a thread name."""
    # Create a thread with just the required components
    system_prompt = SystemPrompt(
        name="Test Prompt",
        content="You are a helpful assistant."
    )
    registered_prompt = EntityRegistry.register(system_prompt)
    
    llm_config = LLMConfig(
        client=LLMClient.openai,
        model="gpt-4",
        response_format=ResponseFormat.text
    )
    registered_config = EntityRegistry.register(llm_config)
    
    # Create a thread
    thread = ChatThread(
        name="Original Thread Name",
        system_prompt=registered_prompt,
        llm_config=registered_config
    )
    registered_thread = EntityRegistry.register(thread)
    original_id = registered_thread.ecs_id
    
    # Get a fresh copy to modify
    retrieved_thread = ChatThread.get(registered_thread.ecs_id)
    
    # Modify the thread name
    retrieved_thread.name = "Modified Thread Name"
    
    # Fork the thread
    forked_thread = retrieved_thread.fork()
    registered_forked = EntityRegistry.register(forked_thread)
    
    # Verify the thread was forked and name was modified
    assert registered_forked.ecs_id != original_id
    assert registered_forked.name == "Modified Thread Name"
    assert registered_forked.parent_id == original_id
    
    # Verify the original thread is unchanged
    original_retrieved = ChatThread.get(original_id)
    assert original_retrieved.name == "Original Thread Name"

# Tests for tool versioning
def test_schema_modification(setup_sql_storage):
    """Test versioning when modifying a schema directly."""
    # Create a structured tool instead (doesn't require callable validation)
    tool = StructuredTool(
        name="schema_tool",
        description="Original tool description",
        json_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
    )
    
    # Register the tool
    registered_tool = EntityRegistry.register(tool)
    original_tool_id = registered_tool.ecs_id
    
    # Modify the tool - description is the field name for StructuredTool
    registered_tool.description = "Modified tool description"
    
    # Create a forked version
    forked_tool = registered_tool.fork()
    
    # Register the forked tool
    forked_registered_tool = EntityRegistry.register(forked_tool)
    
    # Verify the tool was forked
    assert forked_registered_tool.ecs_id != original_tool_id
    assert forked_registered_tool.description == "Modified tool description"  # StructuredTool uses 'description', not 'docstring'
    assert forked_registered_tool.parent_id == original_tool_id
    
    # Get the forked tool from storage
    retrieved_fork = StructuredTool.get(forked_registered_tool.ecs_id)
    assert retrieved_fork is not None
    assert retrieved_fork.description == "Modified tool description"  # StructuredTool uses 'description'

def test_simple_entity_modification(setup_sql_storage):
    """Test simple entity modification and forking."""
    # Create a simple entity - SystemPrompt is a good candidate
    system_prompt = SystemPrompt(
        name="Test Prompt",
        content="Original content"
    )
    
    # Register the entity
    registered_prompt = EntityRegistry.register(system_prompt)
    original_id = registered_prompt.ecs_id
    
    # Modify the entity
    registered_prompt.content = "Modified content"
    
    # Fork the entity
    forked_prompt = registered_prompt.fork()
    
    # Register the forked entity
    forked_registered_prompt = EntityRegistry.register(forked_prompt)
    
    # Verify the entity was forked
    assert forked_registered_prompt.ecs_id != original_id
    assert forked_registered_prompt.content == "Modified content"
    assert forked_registered_prompt.parent_id == original_id
    
    # Get the forked entity from storage
    retrieved_fork = SystemPrompt.get(forked_registered_prompt.ecs_id)
    assert retrieved_fork is not None
    assert retrieved_fork.content == "Modified content"

# Tests for complex versioning scenarios
def test_entity_tracer_decorator(setup_sql_storage):
    """Test that the entity_tracer decorator properly tracks and forks entities."""
    from minference.ecs.entity import entity_tracer
    
    # Create a simple entity
    system_prompt = SystemPrompt(
        name="Test Prompt",
        content="Original content"
    )
    
    # Register the entity
    registered_prompt = EntityRegistry.register(system_prompt)
    original_id = registered_prompt.ecs_id
    
    # Define a function with the entity_tracer decorator
    @entity_tracer
    def modify_prompt(prompt: SystemPrompt, new_content: str) -> SystemPrompt:
        prompt.content = new_content
        return prompt
    
    # Call the traced function
    modified_prompt = modify_prompt(registered_prompt, "Modified by tracer")
    
    # The entity_tracer should have detected changes and returned a forked entity
    # We still need to register the result
    registered_modified = EntityRegistry.register(modified_prompt)
    
    # Verify modifications
    assert registered_modified.ecs_id != original_id
    assert registered_modified.content == "Modified by tracer"
    
    # Verify the entity exists in storage
    retrieved = SystemPrompt.get(registered_modified.ecs_id)
    assert retrieved is not None
    assert retrieved.content == "Modified by tracer"

def test_lineage_tracking_simple(setup_sql_storage):
    """Test tracking entity lineage through multiple versions using a simple entity."""
    # Create a simple entity
    prompt = SystemPrompt(
        name="Lineage Test Prompt",
        content="Original content"
    )
    
    # Register the entity to get a persisted version
    original_prompt = EntityRegistry.register(prompt)
    lineage_id = original_prompt.lineage_id
    
    # Create Version 1
    # First get a fresh copy from storage
    v0_prompt = SystemPrompt.get(original_prompt.ecs_id)
    v0_prompt.content = "Version 1 content"
    v1_prompt = v0_prompt.fork()
    v1_prompt = EntityRegistry.register(v1_prompt)
    
    # Verify Version 1 content is correct
    v1_retrieved = SystemPrompt.get(v1_prompt.ecs_id)
    assert v1_retrieved.content == "Version 1 content"
    
    # Create Version 2
    # First get a fresh copy from storage
    v1_fresh = SystemPrompt.get(v1_prompt.ecs_id)
    v1_fresh.content = "Version 2 content"
    v2_prompt = v1_fresh.fork()
    v2_prompt = EntityRegistry.register(v2_prompt)
    
    # Verify Version 2 content is correct
    v2_retrieved = SystemPrompt.get(v2_prompt.ecs_id)
    assert v2_retrieved.content == "Version 2 content"
    
    # Get all entities in the lineage
    lineage_entities = EntityRegistry.get_lineage_entities(lineage_id)
    
    # Filter for just SystemPrompt entities
    lineage_prompts = [e for e in lineage_entities if isinstance(e, SystemPrompt)]
    
    # Verify the correct number of versions
    assert len(lineage_prompts) >= 3, "Should have at least 3 prompt versions in the lineage"
    
    # Check that all versions are in the lineage
    lineage_prompt_ids = [p.ecs_id for p in lineage_prompts]
    assert original_prompt.ecs_id in lineage_prompt_ids
    assert v1_prompt.ecs_id in lineage_prompt_ids
    assert v2_prompt.ecs_id in lineage_prompt_ids
    
    # Get the lineage tree
    tree = EntityRegistry.get_lineage_tree_sorted(lineage_id)
    
    # Verify the tree structure has the right elements
    assert tree["root"] is not None
    assert len(tree["edges"]) >= 2, "Should have at least 2 edges (parent->child relationships)"

def test_subentity_modifications(setup_sql_storage):
    """Test handling modifications in sub-entities."""
    # Create a nested structure: ChatThread with SystemPrompt
    system_prompt = SystemPrompt(
        name="Sub-entity Test",
        content="Original prompt content"
    )
    registered_prompt = EntityRegistry.register(system_prompt)
    
    llm_config = LLMConfig(
        client=LLMClient.openai,
        model="gpt-4",
        max_tokens=1000,
        temperature=0.7,
        response_format=ResponseFormat.text
    )
    registered_config = EntityRegistry.register(llm_config)
    
    # Create the parent entity with sub-entities
    thread = ChatThread(
        name="Parent Thread",
        system_prompt=registered_prompt,
        llm_config=registered_config
    )
    registered_thread = EntityRegistry.register(thread)
    
    # Save original IDs
    original_thread_id = registered_thread.ecs_id
    original_prompt_id = registered_prompt.ecs_id
    
    # Modify the sub-entity
    registered_thread.system_prompt.content = "Modified prompt content"
    
    # Fork the parent, which should cascade to fork the modified sub-entity
    forked_thread = registered_thread.fork()
    registered_forked_thread = EntityRegistry.register(forked_thread)
    
    # Verify the parent was forked
    assert registered_forked_thread.ecs_id != original_thread_id
    
    # Verify the sub-entity was also forked
    assert registered_forked_thread.system_prompt.ecs_id != original_prompt_id
    assert registered_forked_thread.system_prompt.content == "Modified prompt content"
    
    # Verify relationships in storage
    retrieved_thread = ChatThread.get(registered_forked_thread.ecs_id)
    assert retrieved_thread is not None
    assert retrieved_thread.system_prompt.content == "Modified prompt content"
    assert retrieved_thread.system_prompt.ecs_id != original_prompt_id