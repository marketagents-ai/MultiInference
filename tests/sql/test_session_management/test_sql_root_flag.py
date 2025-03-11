"""
Test to diagnose how the sql_root flag affects entity registration and duplication.
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

def test_sql_root_flag_effect(setup_sql_storage, session):
    """Test how the sql_root flag affects entity registration."""
    # Create a minimal thread
    llm_config = LLMConfig(
        client=LLMClient.openai,
        model="gpt-4",
        response_format=ResponseFormat.text
    )
    
    # By default, sql_root is True on all entities
    print(f"Default sql_root on LLMConfig: {llm_config.sql_root}")
    
    # Register config first
    registered_config = EntityRegistry.register(llm_config)
    print(f"Registered config: {registered_config.ecs_id}")
    
    # Create and register thread
    thread = ChatThread(
        name="Test Thread",
        llm_config=registered_config
    )
    print(f"Default sql_root on Thread: {thread.sql_root}")
    
    # Register the thread
    registered_thread = EntityRegistry.register(thread)
    print(f"Registered thread: {registered_thread.ecs_id}")
    
    # Count SQL entities before adding any messages
    initial_thread_count = session.query(ChatThreadSQL).count()
    initial_message_count = session.query(ChatMessageSQL).count()
    print(f"Initial thread count: {initial_thread_count}")
    print(f"Initial message count: {initial_message_count}")
    
    print("\n=== SCENARIO 1: Messages with sql_root=True (DEFAULT) ===")
    # Create multiple messages with sql_root=True (default)
    messages_with_sql_root = []
    for i in range(3):
        message = ChatMessage(
            role=MessageRole.user if i % 2 == 0 else MessageRole.assistant,
            content=f"Message {i+1} with sql_root=True",
            chat_thread_id=registered_thread.ecs_id,
            # Default sql_root=True
        )
        print(f"Message {i+1} sql_root: {message.sql_root}")
        registered_message = EntityRegistry.register(message)
        messages_with_sql_root.append(registered_message)
        print(f"Registered message {i+1}: {registered_message.ecs_id}")
    
    # Count after adding messages with sql_root=True
    count_after_true = session.query(ChatMessageSQL).count()
    print(f"Count after adding 3 messages with sql_root=True: {count_after_true}")
    
    # Get all messages for thread 1 to check their thread_id references
    thread1_messages_before = session.query(ChatMessageSQL).filter(
        ChatMessageSQL.chat_thread_id == registered_thread.ecs_id
    ).all()
    print(f"Thread 1 message count before re-registration: {len(thread1_messages_before)}")
    print("Message IDs and their thread_id values:")
    for msg in thread1_messages_before:
        print(f"  Message {msg.ecs_id} -> thread_id: {msg.chat_thread_id}")
    
    # Update registered_thread.history to include the messages
    # (simulating what happens when a thread naturally accumulates messages)
    registered_thread.history = messages_with_sql_root
    print(f"Updated thread history, now has {len(registered_thread.history)} messages")
    
    # Now register the thread again to see if it re-registers the messages
    print("\nRe-registering thread with messages with sql_root=True...")
    re_registered_thread = EntityRegistry.register(registered_thread)
    print(f"Re-registered thread: {re_registered_thread.ecs_id}")
    
    # Count messages after thread re-registration
    count_after_thread_reregister = session.query(ChatMessageSQL).count()
    print(f"Count after thread re-registration: {count_after_thread_reregister}")
    print(f"Difference: {count_after_thread_reregister - count_after_true} new messages")
    
    # Get all messages for thread 1 after re-registration
    thread1_messages_after = session.query(ChatMessageSQL).filter(
        ChatMessageSQL.chat_thread_id == registered_thread.ecs_id
    ).all()
    print(f"Thread 1 message count after re-registration: {len(thread1_messages_after)}")
    print("Message IDs and their thread_id values:")
    for msg in thread1_messages_after:
        print(f"  Message {msg.ecs_id} -> thread_id: {msg.chat_thread_id}")
    
    # Check for duplicate messages by comparing ecs_ids
    message_ids_before = {msg.ecs_id for msg in thread1_messages_before}
    message_ids_after = {msg.ecs_id for msg in thread1_messages_after}
    new_message_ids = message_ids_after - message_ids_before
    print(f"New message IDs after re-registration: {new_message_ids}")
    
    print("\n=== SCENARIO 2: Messages with sql_root=False ===")
    # Create a fresh thread for comparison
    new_thread = ChatThread(
        name="Test Thread 2",
        llm_config=registered_config
    )
    registered_new_thread = EntityRegistry.register(new_thread)
    print(f"Created new thread: {registered_new_thread.ecs_id}")
    
    # Now create messages with sql_root=False
    messages_without_sql_root = []
    for i in range(3):
        message = ChatMessage(
            role=MessageRole.user if i % 2 == 0 else MessageRole.assistant,
            content=f"Message {i+1} with sql_root=False",
            chat_thread_id=registered_new_thread.ecs_id,
            sql_root=False  # Explicitly set to False
        )
        print(f"Message {i+1} sql_root: {message.sql_root}")
        registered_message = EntityRegistry.register(message)
        messages_without_sql_root.append(registered_message)
        print(f"Registered message {i+1}: {registered_message.ecs_id}")
    
    # Count after adding messages with sql_root=False
    count_after_false = session.query(ChatMessageSQL).count()
    print(f"Count after adding 3 messages with sql_root=False: {count_after_false}")
    
    # Get all messages for thread 2 to check their thread_id references
    thread2_messages_before = session.query(ChatMessageSQL).filter(
        ChatMessageSQL.chat_thread_id == registered_new_thread.ecs_id
    ).all()
    print(f"Thread 2 message count before re-registration: {len(thread2_messages_before)}")
    print("Message IDs and their thread_id values:")
    for msg in thread2_messages_before:
        print(f"  Message {msg.ecs_id} -> thread_id: {msg.chat_thread_id}")
    
    # Update registered_new_thread.history to include the messages
    registered_new_thread.history = messages_without_sql_root
    print(f"Updated thread history, now has {len(registered_new_thread.history)} messages")
    
    # Re-register thread with messages that have sql_root=False
    print("\nRe-registering thread with messages with sql_root=False...")
    re_registered_new_thread = EntityRegistry.register(registered_new_thread)
    print(f"Re-registered thread 2: {re_registered_new_thread.ecs_id}")
    
    # Count messages after second thread re-registration
    final_count = session.query(ChatMessageSQL).count()
    print(f"Final message count: {final_count}")
    print(f"Difference: {final_count - count_after_false} new messages")
    
    # Get all messages for thread 2 after re-registration
    thread2_messages_after = session.query(ChatMessageSQL).filter(
        ChatMessageSQL.chat_thread_id == registered_new_thread.ecs_id
    ).all()
    print(f"Thread 2 message count after re-registration: {len(thread2_messages_after)}")
    print("Message IDs and their thread_id values:")
    for msg in thread2_messages_after:
        print(f"  Message {msg.ecs_id} -> thread_id: {msg.chat_thread_id}")
    
    # Check for duplicate messages by comparing ecs_ids
    message_ids_before2 = {msg.ecs_id for msg in thread2_messages_before}
    message_ids_after2 = {msg.ecs_id for msg in thread2_messages_after}
    new_message_ids2 = message_ids_after2 - message_ids_before2
    print(f"New message IDs after re-registration: {new_message_ids2}")
    
    print("\n=== SUMMARY ===")
    print(f"Thread 1 (sql_root=True): {len(thread1_messages_before)} before, {len(thread1_messages_after)} after")
    print(f"Thread 2 (sql_root=False): {len(thread2_messages_before)} before, {len(thread2_messages_after)} after")
    
    # Check if any duplication occurred
    duplication_occurred_thread1 = len(thread1_messages_after) > len(thread1_messages_before)
    duplication_occurred_thread2 = len(thread2_messages_after) > len(thread2_messages_before)
    
    print(f"Duplication with sql_root=True: {duplication_occurred_thread1}")
    print(f"Duplication with sql_root=False: {duplication_occurred_thread2}")
    
    # The test fails if re-registering a thread with sql_root=True messages creates duplicate messages
    assert not duplication_occurred_thread1, "Re-registering thread with sql_root=True messages created duplicates"
    assert not duplication_occurred_thread2, "Re-registering thread with sql_root=False messages created duplicates"