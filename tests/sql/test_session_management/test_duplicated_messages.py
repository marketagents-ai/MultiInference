"""
Test to diagnose duplication of messages in the SQL storage backend.
"""

import uuid
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, cast, Any
from uuid import UUID

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker, joinedload

from minference.ecs.entity import Entity
from minference.ecs.enregistry import EntityRegistry
from minference.threads.models import (
    ChatMessage, ChatThread, MessageRole, LLMConfig, LLMClient, ResponseFormat
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
    
    # Import the Base from storage.py and sql_models.py to create all tables
    from minference.ecs.storage import BaseEntitySQL, Base as EntityBase_Base
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
    from minference.ecs.storage import SqlEntityStorage
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

def test_message_duplication_diagnosis(setup_sql_storage, session):
    """Test to diagnose why messages are being duplicated in SQL storage."""
    # Create a minimal thread
    llm_config = LLMConfig(
        client=LLMClient.openai,
        model="gpt-4",
        response_format=ResponseFormat.text
    )
    
    # Register config first
    registered_config = EntityRegistry.register(llm_config)
    print(f"Registered config: {registered_config.ecs_id}")
    
    # Create and register thread
    thread = ChatThread(
        name="Test Thread",
        llm_config=registered_config
    )
    
    # Register the thread to save it to SQL storage
    registered_thread = EntityRegistry.register(thread)
    print(f"Registered thread: {registered_thread.ecs_id}")
    
    # Count SQL messages before adding any
    initial_count = session.query(ChatMessageSQL).count()
    print(f"Initial message count in SQL: {initial_count}")
    
    # Create message in the thread
    message = ChatMessage(
        role=MessageRole.user,
        content="Test message",
        chat_thread_id=registered_thread.ecs_id
    )
    
    # Register message
    registered_message = EntityRegistry.register(message)
    print(f"Registered message: {registered_message.ecs_id}")
    
    # Count messages after adding one
    count_after_first = session.query(ChatMessageSQL).count()
    print(f"Message count after first message: {count_after_first}")
    
    # Verify count is as expected
    assert count_after_first == initial_count + 1, "Unexpected message count after registration"
    
    # Now let's retrieve the thread and check its messages
    thread_sql = session.query(ChatThreadSQL).options(
        joinedload(ChatThreadSQL.messages)
    ).filter_by(ecs_id=registered_thread.ecs_id).first()
    
    # Check how many messages are associated with the thread
    print(f"Number of messages in thread relationship: {len(thread_sql.messages)}")
    print(f"Message IDs in thread: {[msg.ecs_id for msg in thread_sql.messages]}")
    
    # Should only have one message
    assert len(thread_sql.messages) == 1, "Thread should have exactly one message"
    
    # Now let's retrieve using Entity.to_entity() method
    retrieved_thread = ChatThread.get(registered_thread.ecs_id)
    print(f"Number of messages in thread entity history: {len(retrieved_thread.history)}")
    
    # Test our theory about the to_entity method causing duplication
    thread_entity = thread_sql.to_entity()
    print(f"Number of messages in thread after to_entity: {len(thread_entity.history)}")
    
    # Now register the thread again to see if it causes duplication
    re_registered_thread = EntityRegistry.register(thread_entity)
    print(f"Re-registered thread: {re_registered_thread.ecs_id}")
    
    # Check message count again
    count_after_second = session.query(ChatMessageSQL).count() 
    print(f"Message count after second registration: {count_after_second}")
    
    # Load thread again to check message relationship
    thread_sql_2 = session.query(ChatThreadSQL).options(
        joinedload(ChatThreadSQL.messages)
    ).filter_by(ecs_id=re_registered_thread.ecs_id).first()
    
    print(f"Number of messages in thread after re-registration: {len(thread_sql_2.messages)}")
    print(f"Message IDs after re-registration: {[msg.ecs_id for msg in thread_sql_2.messages]}")
    
    # Verify we're not duplicating messages
    assert count_after_second <= count_after_first + 1, "Messages were duplicated"
    assert len(thread_sql_2.messages) == 1, "Thread should still have exactly one message"

def test_multi_message_duplication(setup_sql_storage, session):
    """Test with multiple messages to see how duplication might occur in tests."""
    # Create a minimal thread
    llm_config = LLMConfig(
        client=LLMClient.openai,
        model="gpt-4",
        response_format=ResponseFormat.text
    )
    
    # Register config and thread
    registered_config = EntityRegistry.register(llm_config)
    thread = ChatThread(
        name="Test Thread",
        llm_config=registered_config
    )
    registered_thread = EntityRegistry.register(thread)
    
    # Initial count
    initial_count = session.query(ChatMessageSQL).count()
    print(f"Initial message count: {initial_count}")
    
    # Create multiple messages
    messages = []
    for i in range(3):
        message = ChatMessage(
            role=MessageRole.user if i % 2 == 0 else MessageRole.assistant,
            content=f"Message {i+1}",
            chat_thread_id=registered_thread.ecs_id
        )
        registered_message = EntityRegistry.register(message)
        messages.append(registered_message)
        
        # Print registration data
        print(f"Registered message {i+1}: {registered_message.ecs_id}")
    
    # Count after adding messages
    count_after_adding = session.query(ChatMessageSQL).count()
    print(f"Count after adding 3 messages: {count_after_adding}")
    assert count_after_adding == initial_count + 3, "Should have exactly 3 new messages"
    
    # Check thread's messages
    thread_sql = session.query(ChatThreadSQL).options(
        joinedload(ChatThreadSQL.messages)
    ).filter_by(ecs_id=registered_thread.ecs_id).first()
    
    print(f"Messages in thread relationship: {len(thread_sql.messages)}")
    message_ids = [msg.ecs_id for msg in thread_sql.messages]
    print(f"Message IDs: {message_ids}")
    
    # Should have 3 messages
    assert len(thread_sql.messages) == 3, "Thread should have exactly 3 messages"
    
    # Now retrieve using to_entity and register again - this mimics test code paths
    thread_entity = thread_sql.to_entity()
    print(f"Messages in thread entity after to_entity: {len(thread_entity.history)}")
    
    # Check for message duplication in the entity
    entity_message_ids = [msg.ecs_id for msg in thread_entity.history]
    print(f"Entity message IDs: {entity_message_ids}")
    
    # Check for uniqueness
    assert len(entity_message_ids) == len(set(entity_message_ids)), "Duplicate message IDs in entity"
    
    # Re-register the entity
    re_registered = EntityRegistry.register(thread_entity)
    
    # Check SQL again
    thread_sql_2 = session.query(ChatThreadSQL).options(
        joinedload(ChatThreadSQL.messages)
    ).filter_by(ecs_id=re_registered.ecs_id).first()
    
    final_count = len(thread_sql_2.messages)
    print(f"Final message count in thread: {final_count}")
    print(f"Final message IDs: {[msg.ecs_id for msg in thread_sql_2.messages]}")
    
    # This assertion will fail if we have the duplication bug
    assert final_count == 3, f"Expected 3 messages, got {final_count}"