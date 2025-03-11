"""
Tests for the CallableTool entity in the threads module.
"""
import pytest
import json
from typing import cast, Optional, Dict, Any
from uuid import UUID

from minference.ecs.enregistry import EntityRegistry
from minference.ecs.caregistry import CallableRegistry
from minference.threads.models import CallableTool

class TestCallableTool:
    """Tests for the CallableTool entity."""
    
    def test_tool_creation_from_callable(self):
        """Test creation of a CallableTool from a callable function."""
        def test_func(x: int, y: int) -> int:
            """Test function that adds two numbers."""
            return x + y
            
        tool = CallableTool.from_callable(test_func)
        
        # Verify properties
        assert tool.name == "test_func"
        assert tool.docstring == "Test function that adds two numbers."
        assert tool.input_schema is not None
        assert tool.output_schema is not None
        
        # Verify registration in both registries
        # - EntityRegistry for the tool entity itself
        # - CallableRegistry for the function
        retrieved_tool = CallableTool.get(tool.ecs_id)
        assert retrieved_tool is not None
        
        registered_func = CallableRegistry.get(tool.name)
        assert registered_func is not None
    
    def test_tool_from_source(self):
        """Test creation of a CallableTool from source code."""
        source = """
def multiply(x: int, y: int) -> int:
    \"\"\"Multiply two numbers together.\"\"\"
    return x * y
"""
        tool = CallableTool.from_source(source, name="multiply")
        
        # Verify properties
        assert tool.name == "multiply"
        assert tool.docstring == "Multiply two numbers together."
        assert tool.callable_text == source
        
        # Verify function was registered and works
        function = CallableRegistry.get("multiply")
        assert function is not None
        assert function(3, 4) == 12
    
    def test_tool_execution(self, callable_tool):
        """Test executing the tool's function."""
        # Using the callable_tool fixture from conftest.py
        result = callable_tool.execute({"x": 5, "y": 7})
        assert result["result"] == 12
    
    def test_async_tool_execution(self):
        """Test executing an async tool."""
        async def async_add(x: int, y: int) -> int:
            """Add two numbers asynchronously."""
            return x + y
            
        tool = CallableTool.from_callable(async_add)
        
        # Run the test using pytest-asyncio
        import asyncio
        result = asyncio.run(tool.aexecute({"x": 3, "y": 4}))
        assert result["result"] == 7
    
    def test_tool_openai_format(self, callable_tool):
        """Test conversion to OpenAI tool format."""
        openai_tool = callable_tool.get_openai_tool()
        
        # Verify OpenAI tool format
        assert openai_tool is not None
        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == callable_tool.name
        assert openai_tool["function"]["description"] == callable_tool.docstring
        
        # Verify parameters schema
        assert "parameters" in openai_tool["function"]
        assert openai_tool["function"]["parameters"] == callable_tool.input_schema
    
    def test_tool_anthropic_format(self, callable_tool):
        """Test conversion to Anthropic tool format."""
        anthropic_tool = callable_tool.get_anthropic_tool()
        
        # Verify Anthropic tool format
        assert anthropic_tool is not None
        assert anthropic_tool["name"] == callable_tool.name
        assert anthropic_tool["description"] == callable_tool.docstring
        assert anthropic_tool["input_schema"] == callable_tool.input_schema
        assert "cache_control" in anthropic_tool
    
    def test_tool_creation_and_registration(self):
        """Test creating two different tool instances preserves their schemas."""
        # Make sure to include type hints for the function
        def add_func1(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y
            
        def add_func2(x: int, y: int) -> int:
            """Add two integers together."""
            return x + y
            
        tool1 = CallableTool.from_callable(
            add_func1,
            name="add1",
            docstring="Add two numbers"
        )
        
        tool2 = CallableTool.from_callable(
            add_func2,
            name="add2",
            docstring="Add two integers together"
        )
        
        # Verify the tools have different IDs
        assert tool1.ecs_id != tool2.ecs_id
        
        # Schemas may differ in title but should have same structure
        assert tool1.input_schema['properties'] == tool2.input_schema['properties']
        assert tool1.input_schema['required'] == tool2.input_schema['required']
        assert tool1.output_schema['properties'] == tool2.output_schema['properties']
        
        # Verify the docstrings are different
        assert tool1.docstring == "Add two numbers"
        assert tool2.docstring == "Add two integers together"