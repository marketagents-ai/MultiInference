"""
Tests for basic entity conversion between domain entities and SQL models.

These tests verify that entities can be properly converted to their corresponding
SQLAlchemy ORM models and back with proper field mapping.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, cast, Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from minference.threads.models import (
    SystemPrompt, LLMConfig, LLMClient, ResponseFormat 
)

from minference.threads.sql_models import (
    Base, SystemPromptSQL, LLMConfigSQL
)

# Setup SQLite in-memory database for testing
@pytest.fixture
def engine():
    """Create an in-memory SQLite engine."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine

@pytest.fixture
def session(engine):
    """Create a database session."""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_system_prompt_identity_preservation(session):
    """Test that SystemPrompt identity is preserved during ORM conversion roundtrip."""
    # Create a SystemPrompt entity
    system_prompt = SystemPrompt(
        name="Assistant Prompt",
        content="You are a helpful assistant."
    )
    
    # Convert to SQL model
    sql_model = SystemPromptSQL.from_entity(system_prompt)
    session.add(sql_model)
    session.commit()
    
    # Query the model back from DB
    queried_sql = session.query(SystemPromptSQL).filter_by(ecs_id=system_prompt.ecs_id).first()
    
    # Convert back to entity
    reconverted_entity = queried_sql.to_entity()
    
    # Check identity preservation
    assert reconverted_entity.ecs_id == system_prompt.ecs_id
    assert reconverted_entity.lineage_id == system_prompt.lineage_id
    assert reconverted_entity.parent_id == system_prompt.parent_id
    
    # Use timezone-aware comparison for created_at
    # SQLite might strip timezone info, so compare naive datetime parts
    assert reconverted_entity.created_at.replace(tzinfo=None) == system_prompt.created_at.replace(tzinfo=None)
    
    # Check content preservation
    assert reconverted_entity.name == system_prompt.name
    assert reconverted_entity.content == system_prompt.content

def test_llm_config_mapping(session):
    """Test that LLMConfig fields are correctly mapped in ORM model."""
    # Create a LLMConfig entity with all fields populated
    llm_config = LLMConfig(
        client=LLMClient.openai,
        model="gpt-4-turbo",
        max_tokens=8192,
        temperature=0.7,
        response_format=ResponseFormat.text,
        use_cache=True,
        reasoner=False,
        reasoning_effort="medium"
    )
    
    # Convert to SQL model
    sql_model = LLMConfigSQL.from_entity(llm_config)
    
    # Test field mapping before DB
    assert sql_model.model == llm_config.model
    assert sql_model.provider_name == llm_config.client.value
    assert sql_model.max_tokens == llm_config.max_tokens
    assert sql_model.temperature == llm_config.temperature
    assert sql_model.response_format == llm_config.response_format.value
    
    # Save to DB
    session.add(sql_model)
    session.commit()
    
    # Retrieve from DB
    retrieved_sql = session.query(LLMConfigSQL).filter_by(ecs_id=llm_config.ecs_id).first()
    
    # Convert back to entity
    reconverted_entity = retrieved_sql.to_entity()
    
    # Check identity preservation
    assert reconverted_entity.ecs_id == llm_config.ecs_id
    
    # Check field mapping after round trip
    assert reconverted_entity.client == llm_config.client
    assert reconverted_entity.model == llm_config.model
    assert reconverted_entity.max_tokens == llm_config.max_tokens
    assert reconverted_entity.temperature == llm_config.temperature
    assert reconverted_entity.response_format == llm_config.response_format