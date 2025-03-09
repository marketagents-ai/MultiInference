"""
Tests for the LLMConfig entity in the threads module.
"""
import pytest
from uuid import UUID
from pydantic import ValidationError

from minference.threads.models import LLMConfig, LLMClient, ResponseFormat
from minference.ecs.entity import EntityRegistry


def test_llm_config_creation():
    """Test basic LLMConfig creation and properties."""
    config = LLMConfig(
        client=LLMClient.openai,
        model="gpt-4",
        max_tokens=100,
        temperature=0.7,
        response_format=ResponseFormat.text,
        use_cache=True
    )
    
    # Check that the config was created with the right values
    assert config.client == LLMClient.openai
    assert config.model == "gpt-4"
    assert config.max_tokens == 100
    assert config.temperature == 0.7
    assert config.response_format == ResponseFormat.text
    assert config.use_cache is True
    assert config.reasoner is False
    assert config.reasoning_effort == "medium"
    
    # Check Entity properties
    assert isinstance(config.ecs_id, UUID)
    assert isinstance(config.live_id, UUID)


def test_llm_config_defaults():
    """Test LLMConfig default values."""
    # Create with minimal required fields
    config = LLMConfig(client=LLMClient.openai)
    
    assert config.client == LLMClient.openai
    assert config.model is None
    assert config.max_tokens == 400
    assert config.temperature == 0
    assert config.response_format == ResponseFormat.text
    assert config.use_cache is True
    assert config.reasoner is False
    assert config.reasoning_effort == "medium"


def test_llm_config_registry_integration():
    """Test LLMConfig registry integration."""
    config = LLMConfig(
        client=LLMClient.openai,
        model="gpt-4"
    )
    
    # Register the config
    result = EntityRegistry.register(config)
    assert result is not None
    
    # Retrieve the config
    retrieved = LLMConfig.get(config.ecs_id)
    assert retrieved is not None
    assert isinstance(retrieved, LLMConfig)
    assert retrieved.client == LLMClient.openai
    assert retrieved.model == "gpt-4"


def test_llm_config_validation():
    """Test LLMConfig validation logic."""
    # Test valid combinations
    valid_configs = [
        {"client": LLMClient.openai, "response_format": ResponseFormat.text},
        {"client": LLMClient.openai, "response_format": ResponseFormat.json_object},
        {"client": LLMClient.openai, "response_format": ResponseFormat.structured_output},
        {"client": LLMClient.anthropic, "response_format": ResponseFormat.text},
        {"client": LLMClient.anthropic, "response_format": ResponseFormat.json_beg},
        {"client": LLMClient.openai, "reasoner": True},
    ]
    
    for config_data in valid_configs:
        config = LLMConfig(**config_data)
        assert config is not None
    
    # Test invalid combinations
    invalid_configs = [
        # anthropic + json_object
        {"client": LLMClient.anthropic, "response_format": ResponseFormat.json_object},
        # anthropic + structured_output
        {"client": LLMClient.anthropic, "response_format": ResponseFormat.structured_output},
        # vllm + json_object
        {"client": LLMClient.vllm, "response_format": ResponseFormat.json_object},
        # reasoner with non-OpenAI client
        {"client": LLMClient.anthropic, "reasoner": True},
    ]
    
    for config_data in invalid_configs:
        with pytest.raises(ValidationError):
            LLMConfig(**config_data)


def test_llm_config_fork():
    """Test forking LLMConfig entity."""
    # Create original entity
    original = LLMConfig(
        client=LLMClient.openai,
        model="gpt-4",
        temperature=0.5
    )
    
    # Register the original version
    registered = EntityRegistry.register(original)
    
    # Create a new entity with modifications but same ID
    modified = LLMConfig(
        ecs_id=original.ecs_id,
        client=LLMClient.openai,
        model="gpt-4",
        temperature=0.7
    )
    
    # Fork should detect the changes
    forked = modified.fork()
    
    # Verify the fork created a new ID
    assert forked.ecs_id != original.ecs_id
    # Verify the forked entity has the updated values
    assert forked.temperature == 0.7