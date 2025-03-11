"""
Tests for workflow functionality in ChatThread model.
"""
import pytest
import json
from typing import List, Dict, Optional, Any, cast
from uuid import UUID

from minference.ecs.enregistry import EntityRegistry, entity_tracer
from minference.threads.models import (
    ChatThread, ChatMessage, MessageRole, SystemPrompt, LLMConfig, 
    CallableTool, StructuredTool, ResponseFormat, ProcessedOutput,
    GeneratedJsonObject, LLMClient
)

from conftest import (
    T_ChatThread, T_ChatMessage, T_CallableTool, T_StructuredTool
)

class TestWorkflows:
    """Tests for workflow functionality."""
    
    @pytest.fixture
    def workflow_thread(self, system_prompt, callable_tool):
        """Create a thread configured for workflow."""
        # Create two different callable tools for the workflow
        add_tool = callable_tool  # From fixture
        
        data_tool = CallableTool.from_callable(
            lambda query: {"results": [{"name": f"Result for {query}", "value": len(query) * 10}]},
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
        
        return thread
    
    def test_workflow_initialization(self, workflow_thread):
        """Test initialization of workflow thread."""
        assert workflow_thread.llm_config.response_format == ResponseFormat.workflow
        assert workflow_thread.workflow_step == 0
        assert len(workflow_thread.tools) == 2
    
    async def test_workflow_step_advancement(self, workflow_thread):
        """Test advancing through workflow steps."""
        # Get the first tool in the workflow
        first_tool = workflow_thread.tools[0]
        assert first_tool is not None
        
        # Create a processed output with the first tool
        tool_json = GeneratedJsonObject(
            name=first_tool.name,
            object={"x": 5, "y": 7},
            tool_call_id="call_abc123"
        )
        
        raw_output = {
            "id": "chatcmpl-xyz",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": first_tool.name,
                                    "arguments": json.dumps({"x": 5, "y": 7})
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                    "index": 0
                }
            ],
            "usage": {
                "prompt_tokens": 60,
                "completion_tokens": 30,
                "total_tokens": 90
            }
        }
        
        processed_output = ProcessedOutput(
            content="",
            json_object=tool_json,
            time_taken=2.0,
            llm_client=LLMClient.openai,
            raw_output={
                "raw_result": raw_output,
                "completion_kwargs": {"model": "gpt-3.5-turbo"},
                "start_time": 1677858240.0,
                "end_time": 1677858242.0,
                "chat_thread_id": workflow_thread.ecs_id,
                "chat_thread_live_id": workflow_thread.live_id,
                "client": LLMClient.openai
            },
            chat_thread_id=workflow_thread.ecs_id,
            chat_thread_live_id=workflow_thread.live_id
        )
        
        # Add a user message first
        workflow_thread.new_message = "Let's start the workflow."
        workflow_thread.add_user_message()
        
        # Initial workflow step
        assert workflow_thread.workflow_step == 0
        
        # Add chat turn with tool execution (now using await since it's an async method)
        parent, assistant = await workflow_thread.add_chat_turn_history(processed_output)
        
        # Verify workflow step advanced
        assert workflow_thread.workflow_step == 1
    
    async def test_sequential_tool_execution(self, workflow_thread):
        """Test executing tools in sequence in a workflow."""
        # Add initial user message
        workflow_thread.new_message = "Process the following data: test query"
        workflow_thread.add_user_message()
        
        # First tool (add_tool) output
        first_tool = workflow_thread.tools[0]
        first_tool_output = ProcessedOutput(
            content="",
            json_object=GeneratedJsonObject(
                name=first_tool.name,
                object={"x": 5, "y": 7},
                tool_call_id="call_1"
            ),
            time_taken=1.0,
            llm_client=LLMClient.openai,
            raw_output={
                "raw_result": {},
                "completion_kwargs": {},
                "start_time": 1000.0,
                "end_time": 1001.0,
                "chat_thread_id": workflow_thread.ecs_id,
                "chat_thread_live_id": workflow_thread.live_id,
                "client": LLMClient.openai
            },
            chat_thread_id=workflow_thread.ecs_id,
            chat_thread_live_id=workflow_thread.live_id
        )
        
        # Execute first tool (now using await since it's an async method)
        await workflow_thread.add_chat_turn_history(first_tool_output)
        assert workflow_thread.workflow_step == 1
        
        # Second tool (data_fetcher) output
        second_tool = workflow_thread.tools[1]
        second_tool_output = ProcessedOutput(
            content="",
            json_object=GeneratedJsonObject(
                name=second_tool.name,
                object={"query": "test query"},
                tool_call_id="call_2"
            ),
            time_taken=1.0,
            llm_client=LLMClient.openai,
            raw_output={
                "raw_result": {},
                "completion_kwargs": {},
                "start_time": 1002.0,
                "end_time": 1003.0,
                "chat_thread_id": workflow_thread.ecs_id,
                "chat_thread_live_id": workflow_thread.live_id,
                "client": LLMClient.openai
            },
            chat_thread_id=workflow_thread.ecs_id,
            chat_thread_live_id=workflow_thread.live_id
        )
        
        # Execute second tool (now using await since it's an async method)
        await workflow_thread.add_chat_turn_history(second_tool_output)
        assert workflow_thread.workflow_step == 2
        
        # Verify the complete history
        history = workflow_thread.history
        assert len(history) == 5  # User + 2 pairs of Assistant/Tool messages
        
        # Verify message sequence
        assert history[0].role == MessageRole.user
        assert history[1].role == MessageRole.assistant
        assert history[1].tool_name == first_tool.name
        assert history[2].role == MessageRole.tool
        assert history[2].tool_name == first_tool.name
        assert history[3].role == MessageRole.assistant
        assert history[3].tool_name == second_tool.name
        assert history[4].role == MessageRole.tool
        assert history[4].tool_name == second_tool.name
    
    def test_reset_workflow(self, workflow_thread):
        """Test resetting the workflow."""
        # Initially at step 0
        assert workflow_thread.workflow_step == 0
        
        # Manually advance the workflow
        workflow_thread.workflow_step = 1
        assert workflow_thread.workflow_step == 1
        
        # Reset the workflow
        workflow_thread.reset_workflow_step()
        assert workflow_thread.workflow_step == 0

class TestAutoTools:
    """Tests for auto tools functionality."""
    
    @pytest.fixture
    def auto_tools_thread(self, system_prompt, callable_tool):
        """Create a thread configured for auto tools."""
        # Create two different callable tools
        add_tool = callable_tool  # From fixture
        
        data_tool = CallableTool.from_callable(
            lambda query: {"results": [{"name": f"Result for {query}", "value": len(query) * 10}]},
            name="data_fetcher",
            docstring="Fetch data based on a query"
        )
        
        # Create an auto_tools config
        auto_tools_config = LLMConfig(
            client=LLMClient.openai,
            model="gpt-3.5-turbo",
            max_tokens=100,
            temperature=0,
            response_format=ResponseFormat.auto_tools
        )
        
        # Create thread with auto_tools config and tools
        thread = ChatThread(
            name="Auto Tools Thread",
            system_prompt=system_prompt,
            llm_config=auto_tools_config,
            tools=[add_tool, data_tool]
        )
        
        return thread
    
    def test_auto_tools_initialization(self, auto_tools_thread):
        """Test initialization of auto tools thread."""
        assert auto_tools_thread.llm_config.response_format == ResponseFormat.auto_tools
        assert len(auto_tools_thread.tools) == 2
    
    async def test_auto_tools_execution(self, auto_tools_thread):
        """Test execution of auto-selected tools."""
        # Add initial user message
        auto_tools_thread.new_message = "What is 5 + 7?"
        auto_tools_thread.add_user_message()
        
        # Tool call for the add_tool
        tool = auto_tools_thread.tools[0]  # add_tool
        tool_output = ProcessedOutput(
            content="",
            json_object=GeneratedJsonObject(
                name=tool.name,
                object={"x": 5, "y": 7},
                tool_call_id="call_auto"
            ),
            time_taken=1.0,
            llm_client=LLMClient.openai,
            raw_output={
                "raw_result": {},
                "completion_kwargs": {},
                "start_time": 1000.0,
                "end_time": 1001.0,
                "chat_thread_id": auto_tools_thread.ecs_id,
                "chat_thread_live_id": auto_tools_thread.live_id,
                "client": LLMClient.openai
            },
            chat_thread_id=auto_tools_thread.ecs_id,
            chat_thread_live_id=auto_tools_thread.live_id
        )
        
        # Execute tool (now using await since it's an async method)
        parent, assistant = await auto_tools_thread.add_chat_turn_history(tool_output)
        
        # Verify message sequence
        history = auto_tools_thread.history
        assert len(history) == 3  # User + Assistant + Tool
        
        assert history[0].role == MessageRole.user
        assert history[1].role == MessageRole.assistant
        assert history[1].tool_name == tool.name
        assert history[2].role == MessageRole.tool
        assert history[2].tool_name == tool.name
        
        # Check tool result
        tool_content = json.loads(history[2].content)
        # The tool might return a dict with the result or just the result
        if isinstance(tool_content, dict) and "result" in tool_content:
            assert tool_content["result"] == 12
        else:
            assert tool_content == 12  # 5 + 7 = 12