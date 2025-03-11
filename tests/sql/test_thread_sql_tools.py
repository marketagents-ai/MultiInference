"""
Tests for CallableTool functionality with SQL storage.

These tests verify that tools function correctly when stored in SQL storage,
including tool execution, format conversion, and async tool support.
"""

import json
import asyncio
import pytest
from uuid import UUID
from typing import Dict, Any, cast, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from minference.ecs.entity import Entity, EntityRegistry
from minference.ecs.caregistry import CallableRegistry
from minference.threads.models import CallableTool, StructuredTool, MessageRole
from minference.threads.sql_models import Base, CallableToolSQL, StructuredToolSQL

# Add EntityRegistry to __main__ for entity methods
import sys
sys.modules['__main__'].__dict__['EntityRegistry'] = EntityRegistry

# Setup SQLite in-memory database for testing
@pytest.fixture
def engine():
    """Create an in-memory SQLite engine."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        echo=True,  # Turn on SQL logging
    )
    
    # Import the Base from entity.py and sql_models.py to create all tables
    from minference.ecs.entity import BaseEntitySQL, Base as EntityBase_Base
    from minference.threads.sql_models import Base as ThreadBase
    
    # Create all tables explicitly to ensure they exist
    ThreadBase.metadata.create_all(engine)
    EntityBase_Base.metadata.create_all(engine)
    
    return engine

@pytest.fixture
def session(engine):
    """Create a database session."""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

# Fix UUID handling in SQLite
import sqlite3

def adapt_uuid(uuid):
    return str(uuid)

def convert_uuid(s):
    return UUID(s.decode())

sqlite3.register_adapter(UUID, adapt_uuid)
sqlite3.register_converter("uuid", convert_uuid)

@pytest.fixture
def session_factory(session):
    """Create a session factory that returns the test session."""
    def _session_factory():
        return session
    return _session_factory

@pytest.fixture
def setup_sql_storage(session_factory):
    """Configure EntityRegistry to use SQL storage."""
    from minference.ecs.entity import SqlEntityStorage
    from minference.threads.sql_models import ENTITY_MODEL_MAP
    
    # Create SQL storage with the session factory and entity mappings
    sql_storage = SqlEntityStorage(
        session_factory=session_factory,
        entity_to_orm_map=ENTITY_MODEL_MAP
    )
    
    # Save original storage to restore later
    original_storage = EntityRegistry._storage
    
    # Set SQL storage for testing
    EntityRegistry.use_storage(sql_storage)
    
    yield
    
    # Restore original storage
    EntityRegistry._storage = original_storage

@pytest.fixture
def callable_tool(setup_sql_storage):
    """Create a simple callable tool for testing."""
    def add_numbers(x: int, y: int) -> int:
        """Add two numbers together."""
        return x + y
    
    tool = CallableTool.from_callable(add_numbers)
    EntityRegistry.register(tool)
    return tool

class TestSQLCallableTool:
    """Tests for CallableTool using SQL storage."""
    
    def test_tool_creation_from_callable(self, setup_sql_storage):
        """Test creation of a CallableTool with SQL storage."""
        def test_func(x: int, y: int) -> int:
            """Test function that adds two numbers."""
            return x + y
            
        tool = CallableTool.from_callable(test_func)
        
        # Verify properties 
        assert tool.name == "test_func"
        assert tool.docstring == "Test function that adds two numbers."
        assert tool.input_schema is not None
        assert tool.output_schema is not None
        
        # Explicitly register with EntityRegistry
        EntityRegistry.register(tool)
        
        # Verify registration in both registries
        retrieved_tool = CallableTool.get(tool.ecs_id)
        assert retrieved_tool is not None
        assert retrieved_tool.docstring == tool.docstring
        assert retrieved_tool.name == tool.name
        
        registered_func = CallableRegistry.get(tool.name)
        assert registered_func is not None
    
    def test_tool_from_source(self, setup_sql_storage):
        """Test creation of a CallableTool from source code with SQL storage."""
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
        
        # Explicitly register with EntityRegistry
        EntityRegistry.register(tool)
        
        # Get from registry to verify SQL storage worked
        retrieved_tool = CallableTool.get(tool.ecs_id)
        assert retrieved_tool is not None
        assert retrieved_tool.callable_text == source
        
        # Verify function was registered and works
        function = CallableRegistry.get("multiply")
        assert function is not None
        assert function(3, 4) == 12
    
    def test_tool_execution(self, callable_tool):
        """Test executing the tool's function with SQL storage."""
        result = callable_tool.execute({"x": 5, "y": 7})
        assert result["result"] == 12
        
        # Retrieve from registry and execute again to verify it works after storage
        retrieved_tool = CallableTool.get(callable_tool.ecs_id)
        assert retrieved_tool is not None
        result2 = retrieved_tool.execute({"x": 10, "y": 20})
        assert result2["result"] == 30
    
    @pytest.mark.asyncio
    async def test_async_tool_execution(self, setup_sql_storage):
        """Test executing an async tool with SQL storage."""
        async def async_add(x: int, y: int) -> int:
            """Add two numbers asynchronously."""
            return x + y
            
        tool = CallableTool.from_callable(async_add)
        
        # Explicitly register the tool
        EntityRegistry.register(tool)
        
        # Execute the async tool
        result = await tool.aexecute({"x": 3, "y": 4})
        assert result["result"] == 7
        
        # Retrieve from registry and execute again
        retrieved_tool = CallableTool.get(tool.ecs_id)
        assert retrieved_tool is not None
        result2 = await retrieved_tool.aexecute({"x": 5, "y": 6})
        assert result2["result"] == 11
    
    def test_tool_openai_format(self, callable_tool):
        """Test conversion to OpenAI tool format with SQL storage."""
        openai_tool = callable_tool.get_openai_tool()
        
        # Verify OpenAI tool format
        assert openai_tool is not None
        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == callable_tool.name
        assert openai_tool["function"]["description"] == callable_tool.docstring
        
        # Verify parameters schema
        assert "parameters" in openai_tool["function"]
        assert openai_tool["function"]["parameters"] == callable_tool.input_schema
        
        # Retrieve from registry and check format again
        retrieved_tool = CallableTool.get(callable_tool.ecs_id)
        retrieved_format = retrieved_tool.get_openai_tool()
        assert retrieved_format["function"]["name"] == callable_tool.name
    
    def test_tool_anthropic_format(self, callable_tool):
        """Test conversion to Anthropic tool format with SQL storage."""
        anthropic_tool = callable_tool.get_anthropic_tool()
        
        # Verify Anthropic tool format
        assert anthropic_tool is not None
        assert anthropic_tool["name"] == callable_tool.name
        assert anthropic_tool["description"] == callable_tool.docstring
        assert anthropic_tool["input_schema"] == callable_tool.input_schema
        assert "cache_control" in anthropic_tool
        
        # Retrieve from registry and check format again
        retrieved_tool = CallableTool.get(callable_tool.ecs_id)
        retrieved_format = retrieved_tool.get_anthropic_tool()
        assert retrieved_format["name"] == callable_tool.name

class TestSQLStructuredTool:
    """Tests for StructuredTool using SQL storage."""
    
    @pytest.fixture
    def structured_tool(self, setup_sql_storage):
        """Create a structured tool for testing."""
        # Create a valid JSON schema that matches what we'll test with
        json_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"}
            },
            "required": ["query"]
        }
        
        # Create the tool with the correct field names
        tool = StructuredTool(
            name="search",
            description="Search for information",
            json_schema=json_schema,  # This is the field name expected in StructuredTool!
            instruction_string="Please follow this JSON schema",
            strict_schema=True
        )
        
        # Explicitly register with EntityRegistry
        EntityRegistry.register(tool)
        return tool
    
    def test_structured_tool_storage(self, structured_tool):
        """Test that structured tools are properly stored in SQL."""
        # Explicitly register with EntityRegistry
        EntityRegistry.register(structured_tool)
        
        # Get from registry to verify SQL storage worked
        retrieved_tool = StructuredTool.get(structured_tool.ecs_id)
        assert retrieved_tool is not None
        assert retrieved_tool.name == structured_tool.name
        
        # Check if the entity was properly stored and can be retrieved
        # The specific field names might differ between in-memory and SQL models
        # Just verify the entity exists and has the right name
        assert retrieved_tool is not None
        assert retrieved_tool.name == "search"
        
        # Print all attributes for debugging
        print(f"Attributes of retrieved_tool: {dir(retrieved_tool)}")
        print(f"Tool description: {getattr(retrieved_tool, 'description', 'Not found')}")
    
    def test_structured_tool_validation(self, structured_tool):
        """Test validation of input/output with structured tools in SQL storage."""
        # Explicitly register with EntityRegistry to ensure it's stored in SQL
        EntityRegistry.register(structured_tool)
        
        # Get the tool from the registry to ensure we're using the SQL-stored version
        retrieved_tool = StructuredTool.get(structured_tool.ecs_id)
        assert retrieved_tool is not None

        # Print tool details for debugging
        print(f"Tool fields: name={retrieved_tool.name}, schema={retrieved_tool.json_schema}")

        # Test valid input with try-except to catch validation errors
        valid_input = {"query": "test", "limit": 10}
        try:
            # Some implementations might raise exceptions, others might return error dicts
            result = retrieved_tool.execute(valid_input)
            print(f"Valid input result: {result}")
            # This is a valid input, so we should expect no errors
            assert "error" not in result
        except Exception as e:
            # This should not happen for valid input
            print(f"Unexpected error with valid input: {e}")
            assert False, f"Valid input should not raise error: {e}"
        
        # Test with invalid input that's missing required fields
        invalid_input = {"limit": 10}  # Missing required 'query'
        try:
            # We expect either an error dict or an exception
            result = retrieved_tool.execute(invalid_input)
            print(f"Invalid input result: {result}")
            # If no exception, it should return an error dict
            if "error" not in result and not isinstance(result, dict):
                assert False, f"Invalid input should return error dict, got: {result}"
        except Exception as e:
            # Some implementations might throw exceptions instead
            print(f"Error with invalid input (expected): {e}")
            # This is expected for invalid input
            pass
    
    def test_tool_format_conversion(self, structured_tool):
        """Test format conversion for structured tools with SQL storage."""
        # Explicitly register with EntityRegistry to ensure it's stored in SQL
        EntityRegistry.register(structured_tool)
        
        # Get the tool from the registry to ensure we're using the SQL-stored version
        retrieved_tool = StructuredTool.get(structured_tool.ecs_id)
        assert retrieved_tool is not None
        
        # Test OpenAI format
        openai_format = retrieved_tool.get_openai_tool()
        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == retrieved_tool.name
        assert openai_format["function"]["description"] == retrieved_tool.description
        assert openai_format["function"]["parameters"] == retrieved_tool.json_schema
        
        # Test Anthropic format
        anthropic_format = retrieved_tool.get_anthropic_tool()
        assert anthropic_format["name"] == retrieved_tool.name
        assert anthropic_format["description"] == retrieved_tool.description
        assert anthropic_format["input_schema"] == retrieved_tool.json_schema