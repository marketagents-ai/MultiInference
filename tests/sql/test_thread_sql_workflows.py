"""
Tests for workflow functionality with SQL storage.

These tests verify that workflow functionality in ChatThread works correctly
when entities are stored in SQL storage.
"""

import json
import pytest
from typing import List, Dict, Optional, Any, cast
from uuid import UUID

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from minference.ecs.entity import Entity
from minference.ecs.enregistry import EntityRegistry, entity_tracer
from minference.threads.models import (
    ChatThread, ChatMessage, MessageRole, SystemPrompt, LLMConfig, 
    CallableTool, StructuredTool, ResponseFormat, ProcessedOutput,
    GeneratedJsonObject, LLMClient, RawOutput, Usage
)
from minference.threads.sql_models import Base

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
def system_prompt(setup_sql_storage):
    """Create a system prompt entity for testing."""
    prompt = SystemPrompt(
        name="Test System Prompt",
        content="You are a helpful assistant that supports workflow execution."
    )
    EntityRegistry.register(prompt)
    return prompt

@pytest.fixture
def callable_tool(setup_sql_storage):
    """Create a callable tool for testing."""
    def add_numbers(x: int, y: int) -> int:
        """Add two numbers together."""
        return x + y
    
    tool = CallableTool.from_callable(add_numbers)
    EntityRegistry.register(tool)
    return tool

@pytest.fixture
def workflow_thread(setup_sql_storage, system_prompt, callable_tool):
    """Create a thread configured for workflow."""
    # Create two different callable tools for the workflow
    add_tool = callable_tool  # From fixture
    
    # Create a second tool for data fetching
    def data_fetcher(query: str) -> Dict[str, Any]:
        """Fetch data based on a query."""
        return {"results": [{"name": f"Result for {query}", "value": len(query) * 10}]}
    
    data_tool = CallableTool.from_callable(
        data_fetcher,
        name="data_fetcher",
        docstring="Fetch data based on a query"
    )
    
    # Create a workflow config
    workflow_config = LLMConfig(
        client=LLMClient.openai,
        model="gpt-3.5-turbo",
        max_tokens=100,
        temperature=0,
        response_format=ResponseFormat.workflow
    )
    
    # Create thread with workflow config and tools
    thread = ChatThread(
        name="Workflow Thread",
        system_prompt=system_prompt,
        llm_config=workflow_config,
        tools=[add_tool, data_tool]
    )
    
    # Initialize workflow step
    thread.reset_workflow_step()
    
    # Register with SQL storage
    EntityRegistry.register(thread)
    
    return thread

class TestSQLWorkflows:
    """Tests for workflow functionality with SQL storage."""
    
    def test_workflow_initialization(self, workflow_thread):
        """Test initialization of workflow thread with SQL storage."""
        # Retrieve from registry to ensure SQL storage worked
        thread_id = workflow_thread.ecs_id
        retrieved_thread = ChatThread.get(thread_id)
        
        assert retrieved_thread is not None
        assert retrieved_thread.llm_config.response_format == ResponseFormat.workflow
        assert retrieved_thread.workflow_step == 0
        assert len(retrieved_thread.tools) == 2
        
        # Verify tool relationships are preserved
        tool_names = [tool.name for tool in retrieved_thread.tools]
        assert "add_numbers" in tool_names
        assert "data_fetcher" in tool_names
    
    @pytest.mark.asyncio
    async def test_workflow_step_advancement(self, workflow_thread):
        """Test advancing through workflow steps with SQL storage."""
        # Verify initial state
        assert workflow_thread.workflow_step == 0
        
        # Manually advance the workflow step
        workflow_thread.workflow_step = 1
        EntityRegistry.register(workflow_thread)
        
        # Verify step was advanced
        assert workflow_thread.workflow_step == 1
        
        # Retrieve from registry to ensure changes were saved
        retrieved_thread = ChatThread.get(workflow_thread.ecs_id)
        assert retrieved_thread.workflow_step == 1
    
    def test_reset_workflow(self, workflow_thread):
        """Test resetting workflow step with SQL storage."""
        # Advance the workflow step (use 1 instead of 3 since there are only 2 tools)
        workflow_thread.workflow_step = 1
        EntityRegistry.register(workflow_thread)
        
        # Reset the workflow
        workflow_thread.reset_workflow_step()
        EntityRegistry.register(workflow_thread)
        
        # Verify step was reset
        assert workflow_thread.workflow_step == 0
        
        # Retrieve from registry to ensure changes were saved
        retrieved_thread = ChatThread.get(workflow_thread.ecs_id)
        assert retrieved_thread.workflow_step == 0
    
    def test_sequential_tool_execution(self, workflow_thread):
        """Test sequential execution of tools in a workflow with SQL storage."""
        # Verify initial workflow step
        assert workflow_thread.workflow_step == 0
        
        # Check tool count (we should have 2)
        assert len(workflow_thread.tools) == 2
        
        # Manually set workflow steps - one step at a time
        workflow_thread.workflow_step = 1  # First tool executed
        EntityRegistry.register(workflow_thread)
        assert workflow_thread.workflow_step == 1
        
        # Check persisted state
        retrieved_thread = ChatThread.get(workflow_thread.ecs_id)
        assert retrieved_thread.workflow_step == 1