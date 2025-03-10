"""
Test the creation of SQLAlchemy ORM tables for thread entities.

This simple test just verifies that the tables can be created and reflects 
the correct structure.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy import inspect

from tests.sql.sql_thread_models import Base, ChatThreadSQL, ENTITY_MODEL_MAP

@pytest.fixture
def engine():
    """Create an in-memory SQLite engine."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine

@pytest.fixture
def session(engine):
    """Create a database session."""
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_tables_created(engine):
    """Test that all tables are created."""
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    
    # Check that expected tables exist
    assert "chat_thread" in table_names
    assert "chat_message" in table_names
    assert "system_prompt" in table_names
    assert "llm_config" in table_names
    assert "tool" in table_names
    assert "callable_tool" in table_names
    assert "structured_tool" in table_names
    assert "usage" in table_names
    assert "generated_json_object" in table_names
    assert "raw_output" in table_names
    assert "processed_output" in table_names
    assert "chat_thread_tools" in table_names
    
    # Check table columns for a few tables
    chat_thread_columns = {col["name"] for col in inspector.get_columns("chat_thread")}
    assert "id" in chat_thread_columns
    assert "ecs_id" in chat_thread_columns
    assert "title" in chat_thread_columns
    assert "thread_metadata" in chat_thread_columns  # Renamed from "metadata"
    
    chat_message_columns = {col["name"] for col in inspector.get_columns("chat_message")}
    assert "role" in chat_message_columns
    assert "content" in chat_message_columns
    assert "message_name" in chat_message_columns  # Renamed from "name"
    assert "message_tool_call_id" in chat_message_columns  # Renamed from "tool_call_id"
    
    tool_columns = {col["name"] for col in inspector.get_columns("tool")}
    assert "name" in tool_columns
    assert "tool_description" in tool_columns  # Renamed from "description"
    assert "tool_parameters_schema" in tool_columns  # Renamed from "parameters_schema"
    
    # Check foreign keys
    chat_message_fks = inspector.get_foreign_keys("chat_message")
    chat_thread_tools_fks = inspector.get_foreign_keys("chat_thread_tools")
    
    # Verify at least one foreign key exists in each table
    assert len(chat_message_fks) > 0
    assert len(chat_thread_tools_fks) > 0
    
def test_entity_model_map():
    """Test that the entity to ORM model mapping is complete."""
    # Check we have mappings for all 10 entity types
    assert len(ENTITY_MODEL_MAP) == 10