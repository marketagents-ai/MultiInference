"""
Tests for the RawOutput and ProcessedOutput entities in the threads module.
"""
import pytest
from uuid import UUID, uuid4
import json
import time

from minference.threads.models import (
    RawOutput, 
    ProcessedOutput, 
    LLMClient, 
    GeneratedJsonObject,
    Usage
)
from minference.ecs.entity import EntityRegistry


def test_raw_output_creation():
    """Test basic RawOutput creation and properties."""
    # Create a simple OpenAI-style raw result
    raw_result = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help you today?"
                },
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
    
    # Create RawOutput
    start_time = time.time() - 1  # 1 second ago
    end_time = time.time()
    thread_id = uuid4()
    thread_live_id = uuid4()
    
    raw_output = RawOutput(
        raw_result=raw_result,
        completion_kwargs={"model": "gpt-4", "temperature": 0.7},
        start_time=start_time,
        end_time=end_time,
        chat_thread_id=thread_id,
        chat_thread_live_id=thread_live_id,
        client=LLMClient.openai
    )
    
    # Check that the output was created with the right values
    assert raw_output.raw_result == raw_result
    assert raw_output.completion_kwargs == {"model": "gpt-4", "temperature": 0.7}
    assert raw_output.start_time == start_time
    assert raw_output.end_time == end_time
    assert raw_output.chat_thread_id == thread_id
    assert raw_output.chat_thread_live_id == thread_live_id
    assert raw_output.client == LLMClient.openai
    
    # Check time_taken property
    assert raw_output.time_taken == end_time - start_time
    
    # Check Entity properties
    assert isinstance(raw_output.ecs_id, UUID)
    assert isinstance(raw_output.live_id, UUID)


def test_raw_output_parsing_openai():
    """Test RawOutput parsing for OpenAI format."""
    # Create a simple OpenAI-style raw result with text
    raw_result = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help you today?"
                },
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
    
    # Create RawOutput
    raw_output = RawOutput(
        raw_result=raw_result,
        completion_kwargs={},
        start_time=time.time() - 1,
        end_time=time.time(),
        chat_thread_id=uuid4(),
        chat_thread_live_id=uuid4(),
        client=LLMClient.openai
    )
    
    # Test the str_content property
    assert raw_output.str_content == "Hello, how can I help you today?"
    
    # Test that no JSON object was found
    assert raw_output.json_object is None
    assert raw_output.contains_object is False
    
    # Test usage extraction
    assert raw_output.usage is not None
    assert raw_output.usage.model == "gpt-4"
    assert raw_output.usage.prompt_tokens == 10
    assert raw_output.usage.completion_tokens == 20
    assert raw_output.usage.total_tokens == 30


def test_raw_output_parsing_openai_tool_call():
    """Test RawOutput parsing for OpenAI tool call format."""
    # Create an OpenAI-style raw result with tool call
    tool_call_result = {
        "id": "chatcmpl-456",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "multiply",
                                "arguments": '{"x": 5, "y": 3}'
                            }
                        }
                    ]
                },
                "index": 0,
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40
        }
    }
    
    # Create RawOutput
    raw_output = RawOutput(
        raw_result=tool_call_result,
        completion_kwargs={},
        start_time=time.time() - 1,
        end_time=time.time(),
        chat_thread_id=uuid4(),
        chat_thread_live_id=uuid4(),
        client=LLMClient.openai
    )
    
    # Test content (should be empty for tool calls)
    assert raw_output.str_content == ""
    
    # Test JSON object extraction
    assert raw_output.json_object is not None
    assert raw_output.contains_object is True
    assert raw_output.json_object.name == "multiply"
    assert raw_output.json_object.object == {"x": 5, "y": 3}
    assert raw_output.json_object.tool_call_id == "call_123"


def test_raw_output_parsing_json_content():
    """Test RawOutput parsing for content with embedded JSON."""
    # Create an OpenAI-style raw result with JSON in content
    json_content_result = {
        "id": "chatcmpl-789",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"result": 42, "message": "Answer found"}'
                },
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 15,
            "total_tokens": 20
        }
    }
    
    # Create RawOutput
    raw_output = RawOutput(
        raw_result=json_content_result,
        completion_kwargs={},
        start_time=time.time() - 1,
        end_time=time.time(),
        chat_thread_id=uuid4(),
        chat_thread_live_id=uuid4(),
        client=LLMClient.openai
    )
    
    # Test JSON object extraction from content
    assert raw_output.json_object is not None
    assert raw_output.contains_object is True
    assert raw_output.json_object.name == "parsed_content"
    assert raw_output.json_object.object == {"result": 42, "message": "Answer found"}
    # Content will be None because it was fully parsed as JSON
    assert raw_output.str_content is None


def test_raw_output_registry_integration():
    """Test RawOutput registry integration."""
    raw_output = RawOutput(
        raw_result={"id": "test", "choices": [{"message": {"content": "test"}}]},
        completion_kwargs={},
        start_time=time.time() - 1,
        end_time=time.time(),
        chat_thread_id=uuid4(),
        chat_thread_live_id=uuid4(),
        client=LLMClient.openai
    )
    
    # Register the output
    result = EntityRegistry.register(raw_output)
    assert result is not None
    
    # Retrieve the output
    retrieved = RawOutput.get(raw_output.ecs_id)
    assert retrieved is not None
    assert isinstance(retrieved, RawOutput)
    assert retrieved.raw_result == raw_output.raw_result


def test_processed_output_creation():
    """Test basic ProcessedOutput creation and properties."""
    # Create necessary components
    thread_id = uuid4()
    thread_live_id = uuid4()
    
    # Create a raw output first
    raw_output = RawOutput(
        raw_result={"id": "test", "choices": [{"message": {"content": "test content"}}]},
        completion_kwargs={},
        start_time=time.time() - 1,
        end_time=time.time(),
        chat_thread_id=thread_id,
        chat_thread_live_id=thread_live_id,
        client=LLMClient.openai
    )
    
    # Register raw output
    EntityRegistry.register(raw_output)
    
    # Create JSON object
    json_object = GeneratedJsonObject(
        name="test_result",
        object={"result": "success"}
    )
    
    # Create usage
    usage = Usage(
        model="gpt-4",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30
    )
    
    # Create ProcessedOutput
    processed_output = ProcessedOutput(
        content="Test content",
        json_object=json_object,
        usage=usage,
        error=None,
        time_taken=0.5,
        llm_client=LLMClient.openai,
        raw_output=raw_output,
        chat_thread_id=thread_id,
        chat_thread_live_id=thread_live_id
    )
    
    # Check that the output was created with the right values
    assert processed_output.content == "Test content"
    assert processed_output.json_object == json_object
    assert processed_output.usage == usage
    assert processed_output.error is None
    assert processed_output.time_taken == 0.5
    assert processed_output.llm_client == LLMClient.openai
    assert processed_output.raw_output == raw_output
    assert processed_output.chat_thread_id == thread_id
    assert processed_output.chat_thread_live_id == thread_live_id
    
    # Check Entity properties
    assert isinstance(processed_output.ecs_id, UUID)
    assert isinstance(processed_output.live_id, UUID)


def test_processed_output_from_raw_output():
    """Test creating ProcessedOutput from RawOutput."""
    # Create thread IDs
    thread_id = uuid4()
    thread_live_id = uuid4()
    
    # Create a raw output with OpenAI format
    raw_result = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help you today?"
                },
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
    
    raw_output = RawOutput(
        raw_result=raw_result,
        completion_kwargs={},
        start_time=time.time() - 1,
        end_time=time.time(),
        chat_thread_id=thread_id,
        chat_thread_live_id=thread_live_id,
        client=LLMClient.openai
    )
    
    # Register raw output
    EntityRegistry.register(raw_output)
    
    # Create ProcessedOutput from RawOutput
    processed_output = raw_output.create_processed_output()
    
    # Check that the processed output has the correct values
    assert processed_output.content == "Hello, how can I help you today?"
    assert processed_output.json_object is None
    assert processed_output.usage is not None
    assert processed_output.usage.model == "gpt-4"
    assert processed_output.error is None
    assert processed_output.time_taken == raw_output.time_taken
    assert processed_output.llm_client == LLMClient.openai
    assert processed_output.raw_output == raw_output
    assert processed_output.chat_thread_id == thread_id
    assert processed_output.chat_thread_live_id == thread_live_id


def test_processed_output_registry_integration():
    """Test ProcessedOutput registry integration."""
    # Create necessary components
    thread_id = uuid4()
    thread_live_id = uuid4()
    
    # Create a raw output
    raw_output = RawOutput(
        raw_result={"id": "test", "choices": [{"message": {"content": "test"}}]},
        completion_kwargs={},
        start_time=time.time() - 1,
        end_time=time.time(),
        chat_thread_id=thread_id,
        chat_thread_live_id=thread_live_id,
        client=LLMClient.openai
    )
    
    # Register raw output
    EntityRegistry.register(raw_output)
    
    # Create ProcessedOutput
    processed_output = ProcessedOutput(
        content="Test content",
        json_object=None,
        usage=None,
        error=None,
        time_taken=0.5,
        llm_client=LLMClient.openai,
        raw_output=raw_output,
        chat_thread_id=thread_id,
        chat_thread_live_id=thread_live_id
    )
    
    # Register the output
    result = EntityRegistry.register(processed_output)
    assert result is not None
    
    # Retrieve the output
    retrieved = ProcessedOutput.get(processed_output.ecs_id)
    assert retrieved is not None
    assert isinstance(retrieved, ProcessedOutput)
    assert retrieved.content == "Test content"
    assert retrieved.llm_client == LLMClient.openai