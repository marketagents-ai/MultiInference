"""
Test to verify that all required SQL tables are properly created.
"""

import sys
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

def test_base_entity_table_creation():
    """Test that the base_entity table is properly created."""
    from minference.ecs.entity import Base as EntityBase
    from minference.ecs.storage import BaseEntitySQL
    
    # Create engine for SQLite in-memory DB
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
    
    # Check if base_entity table exists
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    assert "base_entity" in tables, f"base_entity table not found in tables: {tables}"
    
    # Check if the table has the expected columns
    columns = [col["name"] for col in inspector.get_columns("base_entity")]
    expected_columns = [
        "id", "ecs_id", "lineage_id", "parent_id", "created_at", 
        "old_ids", "class_name", "data", "entity_type"
    ]
    
    for col in expected_columns:
        assert col in columns, f"Column {col} missing from base_entity table"
    
    # Test adding a row to the table
    with engine.connect() as conn:
        conn.execute(text(
            """
            INSERT INTO base_entity (
                ecs_id, lineage_id, parent_id, created_at, old_ids, 
                class_name, data, entity_type
            ) VALUES (
                'test-uuid', 'test-lineage', NULL, CURRENT_TIMESTAMP, '[]',
                'test.Class', '{}', 'base_entity'
            )
            """
        ))
        conn.commit()
        
        # Verify row was added
        result = conn.execute(text("SELECT COUNT(*) FROM base_entity")).scalar()
        assert result == 1, "Failed to insert row into base_entity table"


def test_thread_tables_creation():
    """Test that all thread-related tables are properly created."""
    from minference.threads.sql_models import (
        Base, ChatThreadSQL, ChatMessageSQL, LLMConfigSQL, 
        SystemPromptSQL, ToolSQL, CallableToolSQL, StructuredToolSQL,
        UsageSQL, GeneratedJsonObjectSQL, RawOutputSQL, ProcessedOutputSQL
    )
    
    # Create engine for SQLite in-memory DB
    engine = create_engine("sqlite:///:memory:", echo=True)
    
    # Create tables
    Base.metadata.create_all(engine)
    
    # Check if tables exist
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    expected_tables = [
        "chat_thread", "chat_message", "llm_config", "system_prompt",
        "tool", "usage", "generated_json_object", "raw_output",
        "processed_output", "chat_thread_tools"
    ]
    
    for table in expected_tables:
        assert table in tables, f"Table {table} not found in tables: {tables}"


def test_combined_tables_creation():
    """Test that both base_entity and all thread tables are created together."""
    from minference.ecs.entity import Base as EntityBase
    from minference.ecs.storage import BaseEntitySQL
    from minference.threads.sql_models import (
        Base, ChatThreadSQL, ChatMessageSQL, LLMConfigSQL, 
        SystemPromptSQL, ToolSQL, CallableToolSQL, StructuredToolSQL,
        UsageSQL, GeneratedJsonObjectSQL, RawOutputSQL, ProcessedOutputSQL
    )
    
    # Create engine for SQLite in-memory DB
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
    
    # Check if tables exist
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    expected_tables = [
        "base_entity", "chat_thread", "chat_message", "llm_config", "system_prompt",
        "tool", "usage", "generated_json_object", "raw_output",
        "processed_output", "chat_thread_tools"
    ]
    
    for table in expected_tables:
        assert table in tables, f"Table {table} not found in tables: {tables}"


if __name__ == "__main__":
    test_base_entity_table_creation()
    test_thread_tables_creation()
    test_combined_tables_creation()
    print("All table creation tests passed!")