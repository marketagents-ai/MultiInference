"""
Tests for the Usage entity in the threads module.
"""
import pytest
from uuid import UUID

from minference.threads.models import Usage
from minference.ecs.entity import EntityRegistry


def test_usage_creation():
    """Test basic Usage creation and properties."""
    usage = Usage(
        model="gpt-4",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150
    )
    
    # Check that the usage was created with the right values
    assert usage.model == "gpt-4"
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 150
    
    # Check optional fields have None values
    assert usage.cache_creation_input_tokens is None
    assert usage.cache_read_input_tokens is None
    assert usage.accepted_prediction_tokens is None
    assert usage.audio_tokens is None
    assert usage.reasoning_tokens is None
    assert usage.rejected_prediction_tokens is None
    assert usage.cached_tokens is None
    
    # Check Entity properties
    assert isinstance(usage.ecs_id, UUID)
    assert isinstance(usage.live_id, UUID)


def test_usage_with_optional_fields():
    """Test Usage creation with optional fields."""
    usage = Usage(
        model="claude-3",
        prompt_tokens=200,
        completion_tokens=100,
        total_tokens=300,
        cache_creation_input_tokens=50,
        cache_read_input_tokens=150,
        audio_tokens=10
    )
    
    # Check required fields
    assert usage.model == "claude-3"
    assert usage.prompt_tokens == 200
    assert usage.completion_tokens == 100
    assert usage.total_tokens == 300
    
    # Check optional fields
    assert usage.cache_creation_input_tokens == 50
    assert usage.cache_read_input_tokens == 150
    assert usage.audio_tokens == 10
    
    # Other optional fields should still be None
    assert usage.accepted_prediction_tokens is None
    assert usage.reasoning_tokens is None
    assert usage.rejected_prediction_tokens is None
    assert usage.cached_tokens is None


def test_usage_registry_integration():
    """Test Usage registry integration."""
    usage = Usage(
        model="gpt-4",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150
    )
    
    # Register the usage
    result = EntityRegistry.register(usage)
    assert result is not None
    
    # Retrieve the usage
    retrieved = Usage.get(usage.ecs_id)
    assert retrieved is not None
    assert retrieved.model == "gpt-4"
    assert retrieved.prompt_tokens == 100
    assert retrieved.completion_tokens == 50
    assert retrieved.total_tokens == 150


def test_usage_fork():
    """Test forking Usage entity."""
    # Create original entity
    original = Usage(
        model="gpt-4",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150
    )
    
    # Register the original version
    registered = EntityRegistry.register(original)
    
    # Create a new entity with modifications but same ID
    modified = Usage(
        ecs_id=original.ecs_id,
        model="gpt-4-turbo",
        prompt_tokens=200,
        completion_tokens=50,
        total_tokens=150
    )
    
    # Fork should detect the changes
    forked = modified.fork()
    
    # Verify the fork created a new ID
    assert forked.ecs_id != original.ecs_id
    # Verify the forked entity has the updated values
    assert forked.model == "gpt-4-turbo"
    assert forked.prompt_tokens == 200
    
    # Modify the forked usage further
    forked.completion_tokens = 100
    forked.total_tokens = 300
    
    # Register the forked usage
    EntityRegistry.register(forked)
    
    # Check that both usages exist in the registry with different content
    original_retrieved = Usage.get(original.ecs_id)
    forked_retrieved = Usage.get(forked.ecs_id)
    
    assert original_retrieved is not None
    assert forked_retrieved is not None
    assert original_retrieved.model == "gpt-4"
    assert forked_retrieved.model == "gpt-4-turbo"
    assert original_retrieved.prompt_tokens == 100
    assert forked_retrieved.prompt_tokens == 200
    assert original_retrieved.completion_tokens == 50
    assert forked_retrieved.completion_tokens == 100