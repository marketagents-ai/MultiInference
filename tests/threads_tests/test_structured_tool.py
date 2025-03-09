"""
Tests for the StructuredTool entity in the threads module.
"""
import pytest
import json
from typing import cast, Optional, Dict, Any, List
from uuid import UUID
from pydantic import BaseModel, Field

from minference.ecs.entity import EntityRegistry
from minference.threads.models import StructuredTool

# Define test models for structured output
class TestResponseModel(BaseModel):
    """Test response model with required fields."""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

class ComplexDataModel(BaseModel):
    """Test model with nested structure."""
    name: str
    age: int
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TestStructuredTool:
    """Tests for the StructuredTool entity."""
    
    def test_tool_creation_from_pydantic(self):
        """Test creation of a StructuredTool from a Pydantic model."""
        tool = StructuredTool.from_pydantic(
            TestResponseModel,
            name="test_response",
            description="Generate a test response"
        )
        
        # Verify properties
        assert tool.name == "test_response"
        assert tool.description == "Generate a test response"
        assert tool.json_schema is not None
        assert tool.instruction_string == "Please follow this JSON schema for your response:"
        
        # Verify schema content
        assert "properties" in tool.json_schema
        assert "status" in tool.json_schema["properties"]
        assert "message" in tool.json_schema["properties"]
        assert "data" in tool.json_schema["properties"]
        assert "required" in tool.json_schema
        assert "status" in tool.json_schema["required"]
        assert "message" in tool.json_schema["required"]
        
        # Verify registration in EntityRegistry
        retrieved_tool = StructuredTool.get(tool.ecs_id)
        assert isinstance(retrieved_tool, StructuredTool)
        assert retrieved_tool is not None
        assert retrieved_tool.name == "test_response"
    
    def test_tool_creation_with_custom_params(self):
        """Test creation with custom instruction string."""
        tool = StructuredTool.from_pydantic(
            TestResponseModel,
            name="custom_response",
            description="Custom response generator",
            instruction_string="Use this schema to structure your output:",
            strict_schema=False
        )
        
        # Verify custom properties
        assert tool.name == "custom_response"
        assert tool.description == "Custom response generator"
        assert tool.instruction_string == "Use this schema to structure your output:"
        assert tool.strict_schema is False
    
    def test_tool_validation_success(self):
        """Test successful validation of input against schema."""
        tool = StructuredTool.from_pydantic(TestResponseModel)
        
        # Valid input with all required fields
        valid_input = {
            "status": "success",
            "message": "Operation completed",
            "data": {"key": "value"}
        }
        
        result = tool.execute(valid_input)
        
        # Should return the same data when valid
        assert result == valid_input
    
    def test_tool_validation_failure(self):
        """Test validation failure cases."""
        tool = StructuredTool.from_pydantic(TestResponseModel)
        
        # Missing required field
        invalid_input_1 = {
            "status": "success"
            # missing 'message' field
        }
        
        result_1 = tool.execute(invalid_input_1)
        assert "error" in result_1
        
        # Wrong type for a field
        invalid_input_2 = {
            "status": 123,  # should be string
            "message": "Test"
        }
        
        result_2 = tool.execute(invalid_input_2)
        assert "error" in result_2
    
    def test_complex_schema_validation(self):
        """Test validation with a more complex nested schema."""
        tool = StructuredTool.from_pydantic(ComplexDataModel)
        
        # Valid complex input
        valid_complex = {
            "name": "Test User",
            "age": 30,
            "tags": ["tag1", "tag2"],
            "metadata": {
                "created_at": "2023-01-01",
                "nested": {
                    "key": "value"
                }
            }
        }
        
        result = tool.execute(valid_complex)
        assert result == valid_complex
        
        # Invalid complex input (wrong type in nested structure)
        invalid_complex = {
            "name": "Test User",
            "age": 30,
            "tags": "not_a_list",  # should be a list
            "metadata": {
                "created_at": "2023-01-01"
            }
        }
        
        result = tool.execute(invalid_complex)
        assert "error" in result
    
    def test_schema_instruction(self):
        """Test the formatted schema instruction property."""
        tool = StructuredTool.from_pydantic(
            TestResponseModel,
            instruction_string="Format your response like this:"
        )
        
        instruction = tool.schema_instruction
        
        # Should contain both the instruction and the schema
        assert instruction.startswith("Format your response like this:")
        assert "schema" in instruction.lower() or "properties" in instruction
        
    def test_openai_tool_format(self):
        """Test conversion to OpenAI tool format."""
        tool = StructuredTool.from_pydantic(TestResponseModel)
        
        openai_tool = tool.get_openai_tool()
        
        # Verify OpenAI format
        assert openai_tool is not None
        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == tool.name
        
        assert openai_tool["function"]["description"] == tool.description
        assert openai_tool["function"]["parameters"] == tool.json_schema
    
    def test_anthropic_tool_format(self):
        """Test conversion to Anthropic tool format."""
        tool = StructuredTool.from_pydantic(TestResponseModel)
        
        # With cache
        anthropic_tool_with_cache = tool.get_anthropic_tool(use_cache=True)
        assert anthropic_tool_with_cache is not None
        assert anthropic_tool_with_cache["name"] == tool.name
        assert anthropic_tool_with_cache["description"] == tool.description
        assert anthropic_tool_with_cache["input_schema"] == tool.json_schema
        assert "cache_control" in anthropic_tool_with_cache
        
        # Without cache
        anthropic_tool_no_cache = tool.get_anthropic_tool(use_cache=False)
        assert anthropic_tool_no_cache is not None
        assert "cache_control" not in anthropic_tool_no_cache
    
    def test_openai_json_schema_response(self):
        """Test OpenAI JSON schema response format."""
        tool = StructuredTool.from_pydantic(TestResponseModel)
        
        schema_response = tool.get_openai_json_schema_response()
        
        # Verify response format
        assert schema_response is not None
        assert schema_response["type"] == "json_schema"
        assert schema_response["json_schema"]["name"] == tool.name
        assert schema_response["json_schema"]["description"] == tool.description
        assert schema_response["json_schema"]["schema"] == tool.json_schema
        assert schema_response["json_schema"]["strict"] == tool.strict_schema