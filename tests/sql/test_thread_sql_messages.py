"""
Tests for message relationships in the SQL storage backend.

These tests verify that message relationships are properly maintained in SQL storage.
"""

import uuid
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, cast, Any
from uuid import UUID

import pytest
from sqlalchemy import create_engine, select, Integer, String, JSON, DateTime
from sqlalchemy.orm import Session, sessionmaker, joinedload, declarative_base, mapped_column

from minference.ecs.entity import Entity, EntityRegistry
from minference.threads.models import (
    ChatMessage, ChatThread, MessageRole, Usage, LLMConfig, LLMClient, ResponseFormat
)
from minference.threads.sql_models import (
    Base, ChatMessageSQL, ChatThreadSQL, UsageSQL, ToolSQL
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
    from minference.threads.sql_models import Base as ThreadBase, EntityBase
    
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

@pytest.fixture
def base_chat_thread(setup_sql_storage):
    """Create a basic ChatThread to use as a parent for messages."""
    # Create a minimal thread
    llm_config = LLMConfig(
        client=LLMClient.openai,
        model="gpt-4",
        response_format=ResponseFormat.text
    )
    
    # Register config first
    registered_config = EntityRegistry.register(llm_config)
    
    # Create and register thread
    thread = ChatThread(
        name="Test Thread",
        llm_config=registered_config
    )
    
    # Register the thread to save it to SQL storage
    registered_thread = EntityRegistry.register(thread)
    return registered_thread

def test_basic_message_creation_and_retrieval(setup_sql_storage):
    """Test creating a ChatMessage, saving to SQL, and retrieving it."""
    # Create message entity
    message = ChatMessage(
        role=MessageRole.user,
        content="Hello, this is a test message."
    )
    
    # Register with EntityRegistry to persist to SQL storage
    registered_message = EntityRegistry.register(message)
    
    # Retrieve the message using its ID
    retrieved_message = ChatMessage.get(registered_message.ecs_id)
    assert retrieved_message is not None
    
    # Verify entity matches original
    assert retrieved_message.role == MessageRole.user
    assert retrieved_message.content == "Hello, this is a test message."
    assert retrieved_message.ecs_id == registered_message.ecs_id
    assert retrieved_message.timestamp is not None
    
    # Verify default values for optional fields
    assert retrieved_message.parent_message_uuid is None
    assert retrieved_message.chat_thread_id is None
    assert retrieved_message.tool_uuid is None

def test_parent_child_relationship(setup_sql_storage):
    """Test parent-child message relationships."""
    # Create parent message
    parent = ChatMessage(
        role=MessageRole.user,
        content="Parent message"
    )
    registered_parent = EntityRegistry.register(parent)
    
    # Create child message referencing parent
    child = ChatMessage(
        role=MessageRole.assistant,
        content="Child message",
        parent_message_uuid=registered_parent.ecs_id
    )
    registered_child = EntityRegistry.register(child)
    
    # Retrieve the child message using EntityRegistry
    retrieved_child = ChatMessage.get(registered_child.ecs_id)
    assert retrieved_child is not None
    
    # Verify relationship field is set correctly
    assert retrieved_child.parent_message_uuid == registered_parent.ecs_id
    
    # Now retrieve the parent message
    retrieved_parent = ChatMessage.get(registered_parent.ecs_id)
    assert retrieved_parent is not None
    
    # Verify content
    assert retrieved_parent.content == "Parent message"
    assert retrieved_child.content == "Child message"
    
    # Creating a session to directly query SQL objects for relationship verification
    session_factory = EntityRegistry._storage._session_factory
    session = session_factory()
    
    # Use SQL query to verify the parent-child relationship in the database
    parent_sql = session.query(ChatMessageSQL).options(
        joinedload(ChatMessageSQL.child_messages)
    ).filter_by(ecs_id=registered_parent.ecs_id).first()
    
    # Verify parent has child in SQL relationship
    assert parent_sql is not None
    assert len(parent_sql.child_messages) == 1
    assert parent_sql.child_messages[0].ecs_id == registered_child.ecs_id
    
    # Verify child has parent in SQL relationship
    child_sql = session.query(ChatMessageSQL).options(
        joinedload(ChatMessageSQL.parent_message)
    ).filter_by(ecs_id=registered_child.ecs_id).first()
    
    assert child_sql is not None
    assert child_sql.parent_message is not None
    assert child_sql.parent_message.ecs_id == registered_parent.ecs_id

def test_message_with_usage(setup_sql_storage):
    """Test message with usage statistics."""
    # Create usage
    usage = Usage(
        model="gpt-4",
        completion_tokens=10,
        prompt_tokens=20,
        total_tokens=30
    )
    registered_usage = EntityRegistry.register(usage)
    
    # Create message with usage
    message = ChatMessage(
        role=MessageRole.assistant,
        content="Response with usage tracking",
        usage=registered_usage
    )
    registered_message = EntityRegistry.register(message)
    
    # Retrieve the message using EntityRegistry
    retrieved_message = ChatMessage.get(registered_message.ecs_id)
    assert retrieved_message is not None
    assert retrieved_message.usage is not None
    
    # Verify usage data
    assert retrieved_message.usage.model == "gpt-4"
    assert retrieved_message.usage.completion_tokens == 10
    assert retrieved_message.usage.prompt_tokens == 20
    assert retrieved_message.usage.total_tokens == 30
    
    # Creating a session to directly query SQL objects for relationship verification
    session_factory = EntityRegistry._storage._session_factory
    session = session_factory()
    
    # Query the SQL database directly to verify the relationship
    message_sql = session.query(ChatMessageSQL).options(
        joinedload(ChatMessageSQL.usage)
    ).filter_by(ecs_id=registered_message.ecs_id).first()
    
    # Verify SQL model has usage relationship
    assert message_sql is not None
    assert message_sql.usage is not None
    assert message_sql.usage.completion_tokens == 10
    assert message_sql.usage.prompt_tokens == 20
    assert message_sql.usage.total_tokens == 30
    assert message_sql.usage.model == "gpt-4"

def test_long_conversation_chain(setup_sql_storage):
    """Test a long chain of messages (conversation thread)."""
    # Create messages in a chain
    messages = []
    parent_id = None
    
    # Create 5 messages in a chain
    for i in range(5):
        message = ChatMessage(
            role=MessageRole.user if i % 2 == 0 else MessageRole.assistant,
            content=f"Message {i + 1}",
            parent_message_uuid=parent_id
        )
        registered_message = EntityRegistry.register(message)
        messages.append(registered_message)
        parent_id = registered_message.ecs_id
    
    # Creating a session to directly query SQL objects for relationship verification
    session_factory = EntityRegistry._storage._session_factory
    session = session_factory()
    
    # Retrieve the last message with parent chain
    last_message_id = messages[-1].ecs_id
    
    # Function to recursively get parent chain
    def get_parent_chain(message_id, chain=None):
        if chain is None:
            chain = []
        
        message = session.query(ChatMessageSQL).options(
            joinedload(ChatMessageSQL.parent_message)
        ).filter_by(ecs_id=message_id).first()
        
        if message:
            chain.append(message)
            if message.parent_message_id:
                return get_parent_chain(message.parent_message_id, chain)
        
        return chain
    
    # Get the full chain from the last message
    message_chain = get_parent_chain(last_message_id)
    
    # Verify chain length
    assert len(message_chain) == 5
    
    # Verify order (newest first since we started from the last message)
    for i, message in enumerate(message_chain):
        assert message.content == f"Message {5 - i}"
    
    # Verify parent-child relationships throughout the chain
    for i, message in enumerate(message_chain[:-1]):  # Skip last message which has no parent
        assert message.parent_message_id == message_chain[i + 1].ecs_id
        
    # Now verify the same using the Entity API (not SQL directly)
    # Get the last message
    last_entity = ChatMessage.get(last_message_id)
    assert last_entity is not None
    
    # Manually traverse the chain using parent_message_uuid
    entity_chain = []
    current = last_entity
    while current:
        entity_chain.append(current)
        if current.parent_message_uuid:
            current = ChatMessage.get(current.parent_message_uuid)
        else:
            current = None
    
    # Verify chain length matches
    assert len(entity_chain) == 5
    
    # Verify order and content
    for i, entity in enumerate(entity_chain):
        assert entity.content == f"Message {5 - i}"

def test_sibling_messages(setup_sql_storage, base_chat_thread):
    """Test multiple responses to the same message (siblings)."""
    # Create parent message in the thread
    parent = ChatMessage(
        role=MessageRole.user,
        content="Parent message",
        chat_thread_id=base_chat_thread.ecs_id
    )
    registered_parent = EntityRegistry.register(parent)
    
    # Create multiple child messages (siblings)
    siblings = []
    for i in range(3):
        child = ChatMessage(
            role=MessageRole.assistant,
            content=f"Sibling {i + 1}",
            parent_message_uuid=registered_parent.ecs_id,
            chat_thread_id=base_chat_thread.ecs_id
        )
        registered_child = EntityRegistry.register(child)
        siblings.append(registered_child)
    
    # Creating a session to directly query SQL objects for relationship verification
    session_factory = EntityRegistry._storage._session_factory
    session = session_factory()
    
    # Query the SQL database directly to verify parent-child relationships
    parent_sql = session.query(ChatMessageSQL).options(
        joinedload(ChatMessageSQL.child_messages)
    ).filter_by(ecs_id=registered_parent.ecs_id).first()
    
    # Verify parent has all children
    assert parent_sql is not None
    assert len(parent_sql.child_messages) == 3
    
    # Get all child_messages as entities directly 
    responses = [child.to_entity() for child in parent_sql.child_messages]
    
    # Verify content for each sibling
    sibling_contents = [f"Sibling {i + 1}" for i in range(3)]
    for response in responses:
        assert response.content in sibling_contents
        # Remove from list to ensure each content appears exactly once
        sibling_contents.remove(response.content)
    
    # Verify all contents were found
    assert len(sibling_contents) == 0
    
    # Also verify using the Entity API
    # Get the parent message
    parent_entity = ChatMessage.get(registered_parent.ecs_id)
    assert parent_entity is not None
    
    # Get all siblings by checking parent_message_uuid
    sibling_entities = []
    for sibling_id in [s.ecs_id for s in siblings]:
        sibling = ChatMessage.get(sibling_id)
        if sibling and sibling.parent_message_uuid == parent_entity.ecs_id:
            sibling_entities.append(sibling)
    
    # Verify we found all siblings
    assert len(sibling_entities) == 3
    
    # Verify content again
    sibling_contents = [f"Sibling {i + 1}" for i in range(3)]
    for sibling in sibling_entities:
        assert sibling.content in sibling_contents
        sibling_contents.remove(sibling.content)
    
    assert len(sibling_contents) == 0

def test_message_role_conversion(setup_sql_storage):
    """Test all message roles convert properly between entity and SQL."""
    # Test all message roles
    for role in MessageRole:
        # Create message with this role
        message = ChatMessage(
            role=role,
            content=f"Message with role {role.value}"
        )
        registered_message = EntityRegistry.register(message)
        
        # Retrieve using EntityRegistry
        entity = ChatMessage.get(registered_message.ecs_id)
        assert entity is not None
        assert entity.role == role  # Entity has enum value
        
        # Creating a session to directly query SQL objects
        session_factory = EntityRegistry._storage._session_factory
        session = session_factory()
        
        # Retrieve SQL model directly
        retrieved = session.query(ChatMessageSQL).filter_by(ecs_id=registered_message.ecs_id).first()
        assert retrieved is not None
        assert retrieved.role == role.value  # SQL has string value

def test_message_thread_relationship(setup_sql_storage, base_chat_thread):
    """Test message relationship to chat thread."""
    # Create messages in the thread
    messages = []
    for i in range(3):
        message = ChatMessage(
            role=MessageRole.user if i % 2 == 0 else MessageRole.assistant,
            content=f"Message {i + 1}",
            chat_thread_id=base_chat_thread.ecs_id
        )
        registered_message = EntityRegistry.register(message)
        messages.append(registered_message)
    
    # Creating a session to directly query SQL objects
    session_factory = EntityRegistry._storage._session_factory
    session = session_factory()
    
    # Retrieve the thread with messages from SQL directly
    thread_sql = session.query(ChatThreadSQL).options(
        joinedload(ChatThreadSQL.messages)
    ).filter_by(ecs_id=base_chat_thread.ecs_id).first()
    
    # Verify thread has all messages
    assert thread_sql is not None
    assert len(thread_sql.messages) == 3
    
    # Convert SQL messages to entities directly
    message_entities = [message_sql.to_entity() for message_sql in thread_sql.messages]
    
    # Verify message contents
    message_contents = [f"Message {i + 1}" for i in range(3)]
    for message in message_entities:
        assert message.content in message_contents
        # Remove from list to ensure each content appears exactly once
        message_contents.remove(message.content)
    
    # Verify all contents were found
    assert len(message_contents) == 0
    
    # Also verify using the EntityRegistry
    # Get the thread entity
    thread_entity = ChatThread.get(base_chat_thread.ecs_id)
    assert thread_entity is not None
    
    # Get each message using EntityRegistry and verify it links to the thread
    for message_id in [msg.ecs_id for msg in messages]:
        message = ChatMessage.get(message_id)
        assert message is not None
        assert message.chat_thread_id == thread_entity.ecs_id