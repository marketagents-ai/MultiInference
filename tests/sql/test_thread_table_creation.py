"""
Test to verify that we can create SQLAlchemy tables from our ORM models.
"""

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from tests.sql.sql_thread_models import Base

@pytest.fixture
def sql_engine():
    """Create in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=True,  # Turn on SQL logging to see what's happening
    )
    
    # Create all tables explicitly
    Base.metadata.create_all(engine)
    
    return engine

def test_table_creation(sql_engine: Engine):
    """Test that all SQLAlchemy tables are created correctly."""
    # Use SQLAlchemy's inspect to verify the tables exist
    inspector = inspect(sql_engine)
    table_names = inspector.get_table_names()
    
    # Verify that our core entity tables were created
    expected_tables = [
        "chat_thread", 
        "chat_message",
        "system_prompt",
        "llm_config",
        "tool",
        "callable_tool",
        "structured_tool",
        "usage",
        "generated_json_object",
        "raw_output",
        "processed_output",
        "chat_thread_tools"  # Association table
    ]
    
    for table in expected_tables:
        assert table in table_names, f"Table {table} was not created"
    
    # Verify some key columns in chat_thread table
    columns = {col["name"] for col in inspector.get_columns("chat_thread")}
    assert "id" in columns
    assert "ecs_id" in columns
    assert "title" in columns
    assert "thread_metadata" in columns
    assert "entity_type" in columns
    
    # Verify some key columns in chat_message table
    columns = {col["name"] for col in inspector.get_columns("chat_message")}
    assert "role" in columns
    assert "content" in columns
    assert "chat_thread_id" in columns
    assert "parent_message_id" in columns
    
    # Verify some key relationships 
    foreign_keys = inspector.get_foreign_keys("chat_message")
    fk_columns = {fk["constrained_columns"][0] for fk in foreign_keys}
    assert "chat_thread_id" in fk_columns
    assert "parent_message_id" in fk_columns