"""
Common fixtures and setup for threads tests.
Provides test tool classes and common setup/teardown functionality.
"""
import pytest
import sys
import logging
from uuid import UUID, uuid4
from typing import List, Dict, Optional, Set, Any, Tuple, Union, cast, TypeVar
from pydantic import Field, BaseModel
from io import StringIO

# Import directly from the main module - we're testing the actual implementation
from minference.ecs.entity import Entity, EntityRegistry, InMemoryEntityStorage, entity_tracer
from minference.ecs.caregistry import CallableRegistry
from minference.threads.models import (
    LLMConfig, LLMClient, ResponseFormat, CallableTool, StructuredTool, ChatMessage, 
    MessageRole, SystemPrompt, ChatThread, Usage, GeneratedJsonObject, RawOutput, ProcessedOutput
)

# Type variables for better type hinting
T_ChatThread = TypeVar('T_ChatThread', bound='ChatThread')
T_ChatMessage = TypeVar('T_ChatMessage', bound='ChatMessage')
T_CallableTool = TypeVar('T_CallableTool', bound='CallableTool')
T_StructuredTool = TypeVar('T_StructuredTool', bound='StructuredTool')

# Test message schemas
class TestResponseSchema(BaseModel):
    """Schema for test responses."""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

# Test tool functions
# Not a test function, just a utility for the test fixture
def _test_function(x: int, y: int) -> int:
    """Test function that adds two numbers."""
    return x + y

def data_fetcher(query: str) -> Dict[str, Any]:
    """Example tool function that pretends to fetch data."""
    return {
        "results": [
            {"name": f"Result for {query}", "value": len(query) * 10}
        ],
        "total": 1
    }

async def async_test_function(x: int, y: int) -> int:
    """Async test function that adds two numbers."""
    return x + y

# ========================================================================
# Fixtures
# ========================================================================

@pytest.fixture(autouse=True)
def setup_registry():
    """Setup and teardown for each test."""
    # Make EntityRegistry available in __main__ for Entity methods
    sys.modules['__main__'].EntityRegistry = EntityRegistry # type: ignore
    
    # Use in-memory storage
    storage = InMemoryEntityStorage()
    EntityRegistry.use_storage(storage)
    
    # Setup CallableRegistry logging
    log_stream = StringIO()
    logger = logging.getLogger("CallableRegistry")
    if not hasattr(CallableRegistry, '_logger'):
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        CallableRegistry._logger = logger
        CallableRegistry._log_stream = log_stream
    
    # Register test functions
    CallableRegistry.register("test_function", _test_function)
    CallableRegistry.register("data_fetcher", data_fetcher)
    CallableRegistry.register("async_test_function", async_test_function)
    
    # Run the test
    yield
    
    # Clean up after test
    EntityRegistry.clear()
    if hasattr(CallableRegistry, '_registry'):
        CallableRegistry._registry.clear()
    if hasattr(CallableRegistry, '_timestamps'):
        CallableRegistry._timestamps.clear()

@pytest.fixture
def callable_tool() -> CallableTool:
    """Create a test callable tool."""
    tool = CallableTool.from_callable(_test_function)
    return tool

@pytest.fixture
def structured_tool() -> StructuredTool:
    """Create a test structured tool."""
    tool = StructuredTool.from_pydantic(
        TestResponseSchema,
        name="test_response",
        description="Generate a test response"
    )
    return tool

@pytest.fixture
def system_prompt() -> SystemPrompt:
    """Create a test system prompt."""
    prompt = SystemPrompt(
        name="test_prompt",
        content="You are a helpful assistant for testing."
    )
    return prompt

@pytest.fixture
def llm_config() -> LLMConfig:
    """Create a test LLM configuration."""
    config = LLMConfig(
        client=LLMClient.openai,
        model="gpt-3.5-turbo",
        max_tokens=100,
        temperature=0,
        response_format=ResponseFormat.text
    )
    return config

@pytest.fixture
def empty_chat_thread(llm_config: LLMConfig, system_prompt: SystemPrompt) -> ChatThread:
    """Create an empty chat thread."""
    thread = ChatThread(
        name="Test Thread",
        system_prompt=system_prompt,
        llm_config=llm_config
    )
    return thread

@pytest.fixture
def chat_thread_with_messages(llm_config: LLMConfig, system_prompt: SystemPrompt) -> ChatThread:
    """Create a chat thread with some messages."""
    thread = ChatThread(
        name="Test Thread",
        system_prompt=system_prompt,
        llm_config=llm_config
    )
    
    # Add user message
    thread.new_message = "Hello, this is a test message."
    thread.add_user_message()
    
    # Add assistant message
    assistant_msg = ChatMessage(
        role=MessageRole.assistant,
        content="Hello! How can I help you with testing today?",
        chat_thread_id=thread.ecs_id,
        parent_message_uuid=thread.history[0].ecs_id
    )
    thread.history.append(assistant_msg)
    
    return thread

@pytest.fixture
def chat_thread_with_tools(
    llm_config: LLMConfig, 
    system_prompt: SystemPrompt,
    callable_tool: CallableTool,
    structured_tool: StructuredTool
) -> ChatThread:
    """Create a chat thread with tools."""
    # Create a config with tool response format
    tool_config = LLMConfig(
        client=LLMClient.openai,
        model="gpt-3.5-turbo",
        max_tokens=100,
        temperature=0,
        response_format=ResponseFormat.tool
    )
    
    thread = ChatThread(
        name="Test Thread with Tools",
        system_prompt=system_prompt,
        llm_config=tool_config,
        tools=[callable_tool, structured_tool]
    )
    
    return thread

@pytest.fixture
def usage() -> Usage:
    """Create a test usage object."""
    usage = Usage(
        model="gpt-3.5-turbo",
        prompt_tokens=50,
        completion_tokens=25,
        total_tokens=75
    )
    return usage

@pytest.fixture
def json_object() -> GeneratedJsonObject:
    """Create a test JSON object."""
    json_obj = GeneratedJsonObject(
        name="test_object",
        object={"status": "success", "message": "Test message"}
    )
    return json_obj

@pytest.fixture
def raw_output(
    chat_thread_with_messages: ChatThread,
    usage: Usage
) -> RawOutput:
    """Create a test raw output."""
    # Sample raw result from OpenAI
    raw_result = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a test response."
                },
                "finish_reason": "stop",
                "index": 0
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 25,
            "total_tokens": 75
        }
    }
    
    output = RawOutput(
        raw_result=raw_result,
        completion_kwargs={"model": "gpt-3.5-turbo"},
        start_time=1677858240.0,
        end_time=1677858242.0,
        chat_thread_id=chat_thread_with_messages.ecs_id,
        chat_thread_live_id=chat_thread_with_messages.live_id,
        client=LLMClient.openai
    )
    
    return output

@pytest.fixture
def processed_output(
    raw_output: RawOutput,
    usage: Usage
) -> ProcessedOutput:
    """Create a test processed output."""
    assert raw_output.chat_thread_id is not None
    assert raw_output.chat_thread_live_id is not None
    output = ProcessedOutput(
        content="This is a test response.",
        usage=usage,
        time_taken=2.0,
        llm_client=LLMClient.openai,
        raw_output=raw_output,
        chat_thread_id=raw_output.chat_thread_id,
        chat_thread_live_id=raw_output.chat_thread_live_id
    )
    
    return output

@pytest.fixture
def tool_processed_output(
    chat_thread_with_tools: ChatThread,
    callable_tool: CallableTool,
    usage: Usage
) -> ProcessedOutput:
    """Create a test processed output with tool calls."""
    # Sample raw result from OpenAI with tool call
    raw_result = {
        "id": "chatcmpl-456",
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
                                "name": "test_function",
                                "arguments": "{\"x\": 5, \"y\": 7}"
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
    
    raw_output = RawOutput(
        raw_result=raw_result,
        completion_kwargs={"model": "gpt-3.5-turbo"},
        start_time=1677858240.0,
        end_time=1677858242.0,
        chat_thread_id=chat_thread_with_tools.ecs_id,
        chat_thread_live_id=chat_thread_with_tools.live_id,
        client=LLMClient.openai
    )
    
    json_object = GeneratedJsonObject(
        name="test_function",
        object={"x": 5, "y": 7},
        tool_call_id="call_abc123"
    )
    assert raw_output.chat_thread_id is not None
    assert raw_output.chat_thread_live_id is not None
    
    output = ProcessedOutput(
        content="",
        json_object=json_object,
        usage=usage,
        time_taken=2.0,
        llm_client=LLMClient.openai,
        raw_output=raw_output,
        chat_thread_id=raw_output.chat_thread_id,
        chat_thread_live_id=raw_output.chat_thread_live_id
    )
    
    return output