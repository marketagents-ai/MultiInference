"""
Test for entity merging across sessions with SQLAlchemy.

This test verifies that our fix for session management issues works correctly.
"""

import sys
import pytest
import uuid
from uuid import UUID
from typing import List, Dict, Any, Optional, cast, Tuple

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from minference.ecs.entity import Entity, Base
from minference.ecs.enregistry import EntityRegistry, entity_tracer
from minference.ecs.storage import SqlEntityStorage
from minference.threads.models import ChatThread, ChatMessage, LLMConfig, LLMClient, MessageRole
from minference.threads.sql_models import ChatThreadSQL, ChatMessageSQL, LLMConfigSQL, ENTITY_MODEL_MAP


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


def test_object_already_attached_to_session(setup_sql_storage):
    """Test the core session management issue with our fix."""
    
    # First, create and register a config
    llm_config = LLMConfig(client=LLMClient.openai, model="gpt-4")
    registered_config = EntityRegistry.register(llm_config)
    
    # Create and register a thread
    thread = ChatThread(name="Test Thread", llm_config=registered_config)
    registered_thread = EntityRegistry.register(thread)
    
    # Add a message to the thread
    registered_thread.new_message = "Test message"
    message = registered_thread.add_user_message()
    
    # Previously this would cause "Object already attached to session" error
    # because the message was created in the context of the registered_thread
    # but we're trying to register it separately
    
    # Use merge_entity for the message to avoid session conflicts
    try:
        registered_message = EntityRegistry.merge_entity(message)
        assert registered_message is not None
        assert registered_message.content == "Test message"
        print("Successfully merged message")
    except Exception as e:
        pytest.fail(f"Failed to merge message: {str(e)}")
    
    # Now merge the thread, which should handle the sub-entity properly
    try:
        updated_thread = EntityRegistry.merge_entity(registered_thread)
        assert updated_thread is not None
        assert len(updated_thread.history) == 1
        print("Successfully merged thread after message")
    except Exception as e:
        pytest.fail(f"Failed to merge thread after message: {str(e)}")
    
    # Verify that we can retrieve both entities
    retrieved_thread = ChatThread.get(updated_thread.ecs_id)
    assert retrieved_thread is not None
    assert len(retrieved_thread.history) == 1
    
    retrieved_message = ChatMessage.get(registered_message.ecs_id)
    assert retrieved_message is not None
    assert retrieved_message.content == "Test message"
    assert retrieved_message.chat_thread_id == registered_thread.ecs_id


def test_merge_entity_api(setup_sql_storage):
    """Test the new merge_entity method in EntityRegistry."""
    
    # Create entities with different values to avoid collisions
    llm_config = LLMConfig(client=LLMClient.openai, model="gpt-3.5-turbo")
    registered_config = EntityRegistry.register(llm_config)
    
    thread = ChatThread(name="Merge API Test Thread", llm_config=registered_config)
    registered_thread = EntityRegistry.register(thread)
    
    # Add a message
    registered_thread.new_message = "Test message for merging API"
    message = registered_thread.add_user_message()
    
    # Use the merge_entity API instead of register
    # This should bypass the session conflict issue
    merged_message = EntityRegistry.merge_entity(message)
    assert merged_message is not None
    assert merged_message.ecs_id == message.ecs_id
    
    # Now use merge_entity on the thread too
    updated_thread = EntityRegistry.merge_entity(registered_thread)
    assert updated_thread is not None
    assert len(updated_thread.history) == 1
    
    # Verify relationships
    retrieved_thread = ChatThread.get(updated_thread.ecs_id)
    assert retrieved_thread is not None
    assert len(retrieved_thread.history) == 1
    assert retrieved_thread.history[0].ecs_id == message.ecs_id
    
    # Verify the chat_thread_id was properly set
    retrieved_message = ChatMessage.get(message.ecs_id)
    assert retrieved_message is not None
    assert retrieved_message.chat_thread_id == registered_thread.ecs_id


@entity_tracer
def process_message(thread: ChatThread, content: str) -> ChatMessage:
    """Test function to simulate a real-world entity traced operation."""
    thread.new_message = content
    message = thread.add_user_message()
    return message


def test_entity_tracer_with_merge(setup_sql_storage):
    """Test that the entity_tracer works with our session management fixes."""
    
    # Create a thread with a different model to avoid collisions
    llm_config = LLMConfig(client=LLMClient.openai, model="gpt-4-32k")
    thread = ChatThread(name="Tracer Test Thread", llm_config=llm_config)
    registered_thread = EntityRegistry.register(thread)
    
    # Use the entity_tracer decorated function
    message = process_message(registered_thread, "Test message via tracer")
    
    # Verify the message was created
    assert message is not None
    assert message.content == "Test message via tracer"
    
    # The entity_tracer should automatically register newly created entities
    retrieved_message = ChatMessage.get(message.ecs_id)
    assert retrieved_message is not None
    assert retrieved_message.content == "Test message via tracer"
    
    # Retrieve the thread to verify the message is linked
    retrieved_thread = ChatThread.get(registered_thread.ecs_id)
    assert retrieved_thread is not None
    assert retrieved_thread.history  # Should have at least one message


def test_automated_sub_entity_registration(setup_sql_storage):
    """
    Test that sub-entities are automatically registered when their parent entity is registered.
    This is a critical use case that was previously broken.
    """
    
    # Create a thread with unique model to avoid collisions
    llm_config = LLMConfig(client=LLMClient.openai, model="claude-3-sonnet-20250219")
    thread = ChatThread(name="Auto-Registration Test", llm_config=llm_config)
    registered_thread = EntityRegistry.register(thread)
    
    # Add multiple messages to the thread
    registered_thread.new_message = "First auto-message"
    message1 = registered_thread.add_user_message()
    
    registered_thread.new_message = "Second auto-message"
    message2 = registered_thread.add_user_message()
    
    # Use merge_entity to avoid session conflicts
    updated_thread = EntityRegistry.merge_entity(registered_thread)
    
    # Verify that both messages are returned from the merged thread
    assert len(updated_thread.history) >= 2
    
    # Try to retrieve the first message
    message1_retrieved = ChatMessage.get(message1.ecs_id)
    assert message1_retrieved is not None
    assert message1_retrieved.content == "First auto-message"
    
    # Try to retrieve the second message
    message2_retrieved = ChatMessage.get(message2.ecs_id)
    assert message2_retrieved is not None
    assert message2_retrieved.content == "Second auto-message"


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
    
    # Run the test
    test_object_already_attached_to_session(None)
    test_merge_entity_api(None)
    test_entity_tracer_with_merge(None)
    test_automated_sub_entity_registration(None)
    
    print("All tests passed when run directly!")