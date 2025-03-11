"""
Test script to demonstrate moving RequestLimits to models.py.

This demonstrates moving the RequestLimits class from inference.py to models.py,
which ensures it's properly defined alongside all other entity classes.
"""

import os
import pytest
from typing import Dict, Type, cast, List, Optional, Literal
import sys
from uuid import UUID
from pydantic import Field

from sqlalchemy import create_engine, inspect, text, select
from sqlalchemy.orm import sessionmaker, Session

# Import Entity and storage systems
from minference.ecs.entity import Entity
from minference.ecs.storage import SqlEntityStorage, EntityBase, BaseEntitySQL
from minference.ecs.enregistry import EntityRegistry

# Import the current version from inference.py for reference
from minference.threads.inference import RequestLimits as InferenceRequestLimits

# Import ENTITY_MODEL_MAP and RequestLimitsSQL
from minference.threads.sql_models import (
    Base, ENTITY_MODEL_MAP, RequestLimitsSQL
)

# Add EntityRegistry to __main__ for entity methods
sys.modules['__main__'].__dict__['EntityRegistry'] = EntityRegistry


# Define RequestLimits in models.py style
class RequestLimits(Entity):
    """
    Configuration for API request limits.
    Moved from inference.py to models.py for better integration.
    """
    max_requests_per_minute: int = Field(
        default=50,
        description="The maximum number of requests per minute for the API"
    )
    max_tokens_per_minute: int = Field(
        default=100000,
        description="The maximum number of tokens per minute for the API"
    )
    provider: Literal["openai", "anthropic", "vllm", "litellm", "openrouter"] = Field(
        default="openai",
        description="The provider this limit applies to"
    )


@pytest.fixture
def db_path():
    """Create a temporary database file path."""
    db_file = "test_move_request_limits.db"
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
    # Create all tables
    Base.metadata.create_all(engine)
    # Explicitly create BaseEntitySQL table if needed
    if not inspect(engine).has_table("base_entity"):
        # Force creation of BaseEntitySQL table
        BaseEntitySQL.__table__.create(engine)
    
    return lambda: sessionmaker(bind=engine)()


def test_request_limits_move_compatibility():
    """Test that our moved RequestLimits is compatible with the original."""
    # Create instances of both
    original = InferenceRequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
    moved = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
    
    # Verify they have the same fields
    assert hasattr(moved, 'max_requests_per_minute')
    assert hasattr(moved, 'max_tokens_per_minute')
    assert hasattr(moved, 'provider')
    
    # Verify the values are the same
    assert moved.max_requests_per_minute == original.max_requests_per_minute
    assert moved.max_tokens_per_minute == original.max_tokens_per_minute
    assert moved.provider == original.provider
    
    # Verify they're both Entities
    assert isinstance(original, Entity)
    assert isinstance(moved, Entity)


def test_moved_request_limits_registration(session_factory, engine):
    """Test registering the moved RequestLimits entity with SQL storage."""
    # Configure entity mapping to use the moved RequestLimits
    entity_to_orm_map = cast(Dict[Type[Entity], Type[EntityBase]], dict(ENTITY_MODEL_MAP))
    entity_to_orm_map[Entity] = cast(Type[EntityBase], BaseEntitySQL)
    
    # Add our moved RequestLimits to the mapping
    entity_to_orm_map[RequestLimits] = RequestLimitsSQL
    
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
        limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
        
        # Register it
        registered_limits = EntityRegistry.register(limits)
        
        # Verify it was registered correctly
        assert registered_limits is not None, "RequestLimits was not registered!"
        assert registered_limits.ecs_id == limits.ecs_id, "Registered RequestLimits has different ecs_id!"
        
        # Verify it's in the database using SQL directly
        with session_factory() as session:
            # Query using SQLAlchemy core
            stmt = select(RequestLimitsSQL).where(RequestLimitsSQL.ecs_id == limits.ecs_id)
            result = session.execute(stmt).scalar_one_or_none()
            
            # Check that we found it
            assert result is not None, "RequestLimits not found in database!"
            assert result.max_requests_per_minute == 500, "Wrong max_requests_per_minute!"
            assert result.max_tokens_per_minute == 200000, "Wrong max_tokens_per_minute!"
    finally:
        # Restore original storage
        EntityRegistry._storage = original_storage


def test_solution_recommendations():
    """Document the recommended changes to fix the issue."""
    recommendations = """
    To fix the RequestLimits entity registration issue:
    
    1. Move RequestLimits class from inference.py to models.py:
       - This keeps all entity definitions in one place
       - Makes it clearer that RequestLimits is an Entity
       - Ensures it's available when building ORM mappings
    
    2. Ensure BaseEntitySQL table is created before attempting to use it:
       - Access BaseEntitySQL.__table__ to ensure it's included in metadata
       - Create tables with Base.metadata.create_all(engine)
       - Verify all tables exist with inspect(engine).get_table_names()
    
    3. Explicitly add Entity->BaseEntitySQL mapping:
       - entity_to_orm_map[Entity] = BaseEntitySQL
       - This provides the fallback for any entity not specifically mapped
    
    4. Ensure ENTITY_MODEL_MAP includes all possible entity types:
       - Verify RequestLimits is in ENTITY_MODEL_MAP
       - Check for other entity types in minference/threads/models.py not included
    """
    
    # Print recommendations to console for documentation
    print(recommendations)
    
    # No actual assertion needed - this test is for documentation
    assert True


if __name__ == "__main__":
    # When run directly, print the solution recommendations
    test_solution_recommendations()