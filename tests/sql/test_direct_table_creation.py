"""
Simple test for direct table creation in SQLite.

This tests creating a table directly, without going through SQLAlchemy metadata.
"""

import sqlite3
import pytest
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, 
    JSON, DateTime, inspect, Text, Uuid, text
)

@pytest.fixture
def db_path():
    return ":memory:"

def test_direct_table_creation(db_path):
    """Test direct table creation in SQLite."""
    # Create a new engine
    engine = create_engine(f"sqlite:///{db_path}")
    
    # Create a table directly with SQL
    conn = engine.connect()
    conn.execute(text("""
    CREATE TABLE base_entity (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ecs_id CHAR(32) NOT NULL,
        lineage_id CHAR(32) NOT NULL,
        parent_id CHAR(32),
        created_at DATETIME NOT NULL,
        old_ids TEXT NOT NULL,
        class_name VARCHAR(255) NOT NULL,
        data TEXT,
        entity_type VARCHAR(50) NOT NULL
    )
    """))
    
    # Verify the table exists
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    assert "base_entity" in tables, f"Table not created. Found tables: {tables}"
    
    # Insert some test data
    conn.execute(text("""
    INSERT INTO base_entity 
    (ecs_id, lineage_id, parent_id, created_at, old_ids, class_name, data, entity_type)
    VALUES (:ecs_id, :lineage_id, :parent_id, :created_at, :old_ids, :class_name, :data, :entity_type)
    """), {
        "ecs_id": "123e4567e89b12d3a456426614174000",
        "lineage_id": "123e4567e89b12d3a456426614174001",
        "parent_id": None,
        "created_at": "2023-01-01 00:00:00",
        "old_ids": "[]",
        "class_name": "minference.threads.models.RequestLimits",
        "data": '{"max_requests_per_minute": 500, "max_tokens_per_minute": 20000}',
        "entity_type": "base_entity"
    })
     
    # Query the data back with column names
    result = conn.execute(text("SELECT id, ecs_id, lineage_id, parent_id, created_at, old_ids, class_name, data, entity_type FROM base_entity")).fetchone()
    
    assert result is not None, "No data returned from query"
    assert result[1] == "123e4567e89b12d3a456426614174000", f"Unexpected ecs_id: {result[1]}"
    # Print all fields to see the order
    print(f"Result columns: {[i for i in range(len(result))]} - Values: {[result[i] for i in range(len(result))]}")
    assert result[6] == "minference.threads.models.RequestLimits", f"Unexpected class_name: {result[6]}"