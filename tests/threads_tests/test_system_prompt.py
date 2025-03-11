"""
Tests for the SystemPrompt entity in the threads module.
"""
import pytest
from uuid import UUID

from minference.threads.models import SystemPrompt
from minference.ecs.enregistry import EntityRegistry


def test_system_prompt_creation():
    """Test basic SystemPrompt creation and properties."""
    prompt = SystemPrompt(
        name="test_prompt",
        content="You are a helpful assistant."
    )
    
    # Check that the prompt was created with the right values
    assert prompt.name == "test_prompt"
    assert prompt.content == "You are a helpful assistant."
    
    # Check Entity properties
    assert isinstance(prompt.ecs_id, UUID)
    assert isinstance(prompt.live_id, UUID)


def test_system_prompt_registry_integration():
    """Test SystemPrompt registry integration."""
    prompt = SystemPrompt(
        name="test_prompt",
        content="You are a helpful assistant."
    )
    
    # Register the prompt
    result = EntityRegistry.register(prompt)
    assert result is not None
    
    # Retrieve the prompt
    retrieved = SystemPrompt.get(prompt.ecs_id)
    assert isinstance(retrieved, SystemPrompt)
    assert retrieved is not None
    assert retrieved.name == "test_prompt"
    assert retrieved.content == "You are a helpful assistant."


def test_system_prompt_fork():
    """Test forking SystemPrompt entity."""
    # Create original entity
    original = SystemPrompt(
        name="original_prompt",
        content="You are a helpful assistant."
    )
    
    # Register the original version
    registered = EntityRegistry.register(original)
    
    # Create a new entity with modifications but same ID
    modified = SystemPrompt(
        ecs_id=original.ecs_id,
        name="original_prompt",
        content="You are a very helpful assistant."
    )
    
    # Fork should detect the changes
    forked = modified.fork()
    
    # Verify the fork created a new ID
    assert forked.ecs_id != original.ecs_id
    # Verify the forked entity has the updated values
    assert forked.content == "You are a very helpful assistant."
    
    # Modify the forked prompt further
    forked.name = "modified_prompt"
    
    # Register the forked prompt
    EntityRegistry.register(forked)
    
    # Check that both prompts exist in the registry with different content
    original_retrieved = SystemPrompt.get(original.ecs_id)
    forked_retrieved = SystemPrompt.get(forked.ecs_id)
    
    assert original_retrieved is not None
    assert forked_retrieved is not None
    assert isinstance(original_retrieved, SystemPrompt)
    assert isinstance(forked_retrieved, SystemPrompt)
    assert original_retrieved.name == "original_prompt"
    assert original_retrieved.content == "You are a helpful assistant."
    assert forked_retrieved.name == "modified_prompt"
    assert forked_retrieved.content == "You are a very helpful assistant."