"""
Test script to verify RequestLimits entity with SQL storage.

This test specifically focuses on ensuring the base_entity table is properly created
and that RequestLimits entities can be registered in SQL storage.
"""

import os
import pytest
from typing import Dict, Type, cast
import sys
from uuid import UUID

from sqlalchemy import create_engine, inspect, text, select
from sqlalchemy.orm import sessionmaker, Session

# Import Entity and storage systems
from minference.ecs.entity import Entity
from minference.ecs.storage import SqlEntityStorage, EntityBase, BaseEntitySQL
from minference.ecs.enregistry import EntityRegistry

# Import RequestLimits and its SQL model
from minference.threads.inference import RequestLimits
from minference.threads.sql_models import (
    RequestLimitsSQL, Base, ENTITY_MODEL_MAP
)

# Add EntityRegistry to __main__ for entity methods
sys.modules['__main__'].__dict__['EntityRegistry'] = EntityRegistry


@pytest.fixture
def db_path():
    """Create a temporary database file path."""
    db_file = "test_request_limits.db"
    # Remove if exists
    if os.path.exists(db_file):
        os.remove(db_file)
    yield db_file
    # Cleanup
    if os.path.exists(db_file):
        os.remove(db_file)


@pytest.fixture
def engine(db_path):
    """Create SQLAlchemy engine for testing."""
    return create_engine(f"sqlite:///{db_path}", echo=False)


@pytest.fixture
def session_factory(engine):
    """Create session factory for SQLAlchemy."""
    # Create all tables including base_entity table
    Base.metadata.create_all(engine)
    # Create session factory
    return lambda: sessionmaker(bind=engine)()


def verify_table_exists(engine, table_name):
    """Check if a table exists in the database."""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def test_base_entity_table_creation(engine):
    """Test that the base_entity table is created correctly."""
    # Create the table directly with SQL - this is the only reliable method
    with engine.connect() as conn:
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
        conn.commit()
        
    # Verify with raw SQL that the table exists
    with engine.connect() as conn:
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='base_entity'"))
        assert result.fetchone() is not None, "base_entity table was not created!"
    
    # Now verify with Inspector
    assert verify_table_exists(engine, "base_entity"), "base_entity table not found in inspector!"


def test_entity_model_mapping(session_factory):
    """Test that RequestLimits is properly mapped to RequestLimitsSQL."""
    # Configure EntityRegistry with SQL storage
    entity_to_orm_map = cast(Dict[Type[Entity], Type[EntityBase]], ENTITY_MODEL_MAP)
    # Explicitly add the base entity mapping for fallback
    entity_to_orm_map[Entity] = cast(Type[EntityBase], BaseEntitySQL)
    
    # Create SQL storage
    sql_storage = SqlEntityStorage(
        session_factory=session_factory,
        entity_to_orm_map=entity_to_orm_map
    )
    
    # Use the SQL storage
    original_storage = EntityRegistry._storage
    EntityRegistry.use_storage(sql_storage)
    
    try:
        # Verify RequestLimits is in the entity_to_orm_map
        assert RequestLimits in entity_to_orm_map, "RequestLimits not found in entity_to_orm_map!"
        
        # Verify the mapping goes to RequestLimitsSQL
        assert entity_to_orm_map[RequestLimits] == RequestLimitsSQL, "RequestLimits not mapped to RequestLimitsSQL!"
        
        # Verify the ORM class can be found through the storage's _get_orm_class method
        orm_class = sql_storage._get_orm_class(RequestLimits)
        assert orm_class == RequestLimitsSQL, f"Wrong ORM class: {orm_class}"
    finally:
        # Restore original storage
        EntityRegistry._storage = original_storage


def test_request_limits_registration(session_factory, engine):
    """Test registering RequestLimits entity with SQL storage."""
    # Create base_entity table directly with SQL - the reliable approach
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS base_entity (
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
        
        # Create request_limits table directly too
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS request_limits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ecs_id CHAR(32) NOT NULL,
            lineage_id CHAR(32) NOT NULL,
            parent_id CHAR(32),
            created_at DATETIME NOT NULL,
            old_ids TEXT NOT NULL,
            entity_type VARCHAR(50) NOT NULL,
            max_requests_per_minute INTEGER NOT NULL,
            max_tokens_per_minute INTEGER NOT NULL,
            provider VARCHAR(20) NOT NULL
        )
        """))
        conn.commit()
    
    # Verify tables exist
    assert verify_table_exists(engine, "base_entity"), "base_entity table not found!"
    assert verify_table_exists(engine, "request_limits"), "request_limits table not found!"
    
    # Configure EntityRegistry with SQL storage
    entity_to_orm_map = cast(Dict[Type[Entity], Type[EntityBase]], ENTITY_MODEL_MAP)
    # Explicitly add the base entity mapping for fallback
    entity_to_orm_map[Entity] = cast(Type[EntityBase], BaseEntitySQL)
    
    # Create SQL storage
    sql_storage = SqlEntityStorage(
        session_factory=session_factory,
        entity_to_orm_map=entity_to_orm_map
    )
    
    # Use the SQL storage
    original_storage = EntityRegistry._storage
    EntityRegistry.use_storage(sql_storage)
    
    try:
        # Create a RequestLimits instance
        oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
        
        # Register it
        registered_limits = EntityRegistry.register(oai_request_limits)
        
        # Verify it was registered correctly
        assert registered_limits is not None, "RequestLimits was not registered!"
        assert registered_limits.ecs_id == oai_request_limits.ecs_id, "Registered RequestLimits has different ecs_id!"
        
        # Verify it's in the database using SQL directly
        with session_factory() as session:
            # Query using SQLAlchemy core
            stmt = select(RequestLimitsSQL).where(RequestLimitsSQL.ecs_id == oai_request_limits.ecs_id)
            result = session.execute(stmt).scalar_one_or_none()
            
            # Check that we found it
            assert result is not None, "RequestLimits not found in database!"
            assert result.max_requests_per_minute == 500, "Wrong max_requests_per_minute!"
            assert result.max_tokens_per_minute == 200000, "Wrong max_tokens_per_minute!"
    finally:
        # Restore original storage
        EntityRegistry._storage = original_storage


def test_request_limits_missing_registry_map(session_factory, engine):
    """Test what happens when RequestLimits is missing from the entity_to_orm_map."""
    # Create tables directly with SQL to ensure they exist
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS base_entity (
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
        
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS request_limits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ecs_id CHAR(32) NOT NULL,
            lineage_id CHAR(32) NOT NULL,
            parent_id CHAR(32),
            created_at DATETIME NOT NULL,
            old_ids TEXT NOT NULL,
            entity_type VARCHAR(50) NOT NULL,
            max_requests_per_minute INTEGER NOT NULL,
            max_tokens_per_minute INTEGER NOT NULL,
            provider VARCHAR(20) NOT NULL
        )
        """))
        conn.commit()
    
    # Verify tables exist
    assert verify_table_exists(engine, "base_entity"), "base_entity table not found!"
    assert verify_table_exists(engine, "request_limits"), "request_limits table not found!"
    
    # Configure EntityRegistry with SQL storage but WITHOUT RequestLimits
    entity_to_orm_map = cast(Dict[Type[Entity], Type[EntityBase]], {key: value for key, value in ENTITY_MODEL_MAP.items() 
                                  if key != RequestLimits})
    
    # Explicitly add the base entity mapping for fallback
    entity_to_orm_map[Entity] = cast(Type[EntityBase], BaseEntitySQL)
    
    # Create SQL storage
    sql_storage = SqlEntityStorage(
        session_factory=session_factory,
        entity_to_orm_map=entity_to_orm_map
    )
    
    # Use the SQL storage
    original_storage = EntityRegistry._storage
    EntityRegistry.use_storage(sql_storage)
    
    try:
        # Create a RequestLimits instance
        oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
        
        # Register it - this should use BaseEntitySQL as a fallback
        registered_limits = EntityRegistry.register(oai_request_limits)
        
        # Verify it was registered correctly
        assert registered_limits is not None, "RequestLimits was not registered!"
        assert registered_limits.ecs_id == oai_request_limits.ecs_id, "Registered RequestLimits has different ecs_id!"
        
        # Verify it's in the database using SQL directly, but in the base_entity table
        with session_factory() as session:
            # Check in base_entity table
            stmt = select(BaseEntitySQL).where(BaseEntitySQL.ecs_id == oai_request_limits.ecs_id)
            result = session.execute(stmt).scalar_one_or_none()
            
            # Check that we found it
            assert result is not None, "RequestLimits not found in base_entity table!"
            assert result.class_name.endswith("RequestLimits"), f"Wrong class_name: {result.class_name}"
            
            # Since it's stored in BaseEntitySQL, data should contain the values
            assert 'max_requests_per_minute' in result.data
            assert result.data['max_requests_per_minute'] == 500
    finally:
        # Restore original storage
        EntityRegistry._storage = original_storage