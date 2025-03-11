"""
Tests for the GeneratedJsonObject entity in the threads module.
"""
import pytest
from uuid import UUID

from minference.threads.models import GeneratedJsonObject
from minference.ecs.enregistry import EntityRegistry


def test_generated_json_object_creation():
    """Test basic GeneratedJsonObject creation and properties."""
    json_obj = GeneratedJsonObject(
        name="test_object",
        object={"key": "value", "number": 42}
    )
    
    # Check that the object was created with the right values
    assert json_obj.name == "test_object"
    assert json_obj.object == {"key": "value", "number": 42}
    assert json_obj.tool_call_id is None
    
    # Check Entity properties
    assert isinstance(json_obj.ecs_id, UUID)
    assert isinstance(json_obj.live_id, UUID)


def test_generated_json_object_with_tool_call_id():
    """Test GeneratedJsonObject with tool_call_id."""
    json_obj = GeneratedJsonObject(
        name="multiply_result",
        object={"result": 15},
        tool_call_id="call_123456"
    )
    
    assert json_obj.name == "multiply_result"
    assert json_obj.object == {"result": 15}
    assert json_obj.tool_call_id == "call_123456"


def test_generated_json_object_complex_structure():
    """Test GeneratedJsonObject with complex nested structure."""
    complex_obj = {
        "person": {
            "name": "John Doe",
            "age": 30,
            "contact": {
                "email": "john@example.com",
                "phone": "555-1234"
            }
        },
        "items": [
            {"id": 1, "name": "Item 1"},
            {"id": 2, "name": "Item 2"},
            {"id": 3, "name": "Item 3"}
        ],
        "active": True
    }
    
    json_obj = GeneratedJsonObject(
        name="person_record",
        object=complex_obj
    )
    
    assert json_obj.name == "person_record"
    assert json_obj.object == complex_obj
    assert json_obj.object["person"]["name"] == "John Doe"
    assert len(json_obj.object["items"]) == 3
    assert json_obj.object["active"] is True


def test_generated_json_object_registry_integration():
    """Test GeneratedJsonObject registry integration."""
    json_obj = GeneratedJsonObject(
        name="test_object",
        object={"key": "value"}
    )
    
    # Register the object
    result = EntityRegistry.register(json_obj)
    assert result is not None
    
    # Retrieve the object
    retrieved = GeneratedJsonObject.get(json_obj.ecs_id)
    assert isinstance(retrieved, GeneratedJsonObject)

    assert retrieved is not None
    assert retrieved.name == "test_object"
    assert retrieved.object == {"key": "value"}


def test_generated_json_object_fork():
    """Test forking GeneratedJsonObject entity."""
    # Create original entity
    original = GeneratedJsonObject(
        name="original_object",
        object={"key": "value"}
    )
    
    # Register the original version
    registered = EntityRegistry.register(original)
    
    # Create a new entity with modifications but same ID
    modified = GeneratedJsonObject(
        ecs_id=original.ecs_id,
        name="original_object",
        object={"key": "updated_value"}
    )
    
    # Fork should detect the changes
    forked = modified.fork()
    
    # Verify the fork created a new ID
    assert forked.ecs_id != original.ecs_id
    # Verify the forked entity has the updated values
    assert forked.object == {"key": "updated_value"}
    
    # Modify the forked object further
    forked.name = "modified_object"
    forked.object = {"key": "new_value", "additional": True}
    
    # Register the forked object
    EntityRegistry.register(forked)
    
    # Check that both objects exist in the registry with different content
    original_retrieved = GeneratedJsonObject.get(original.ecs_id)
    assert isinstance(original_retrieved, GeneratedJsonObject)
    forked_retrieved = GeneratedJsonObject.get(forked.ecs_id)
    assert isinstance(forked_retrieved, GeneratedJsonObject)
    assert original_retrieved is not None
    assert forked_retrieved is not None
    assert original_retrieved.name == "original_object"
    assert original_retrieved.object == {"key": "value"}
    assert forked_retrieved.name == "modified_object"
    assert forked_retrieved.object == {"key": "new_value", "additional": True}