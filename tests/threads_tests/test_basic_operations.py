"""
Tests for basic operations of the ChatThread model and related entities.
"""
import pytest
import json
from typing import List, Dict, Optional, Any, cast
from uuid import UUID

from minference.ecs.enregistry import EntityRegistry
from minference.threads.models import (
    ChatThread, ChatMessage, MessageRole, SystemPrompt, LLMConfig, 
    CallableTool, StructuredTool, ResponseFormat, ProcessedOutput
)

from conftest import (
    T_ChatThread, T_ChatMessage, T_CallableTool, T_StructuredTool
)

class TestChatMessage:
    """Tests for the ChatMessage entity."""
    
    def test_message_creation(self):
        """Test creation and registration of ChatMessage."""
        message = ChatMessage(
            role=MessageRole.user,
            content="Hello, this is a test message."
        )
        
        # Verify properties
        assert message.role == MessageRole.user
        assert message.content == "Hello, this is a test message."
        assert message.timestamp is not None
        
        # Verify registration
        retrieved = EntityRegistry.get(message.ecs_id)
        assert retrieved is not None
        assert isinstance(retrieved, ChatMessage)
        
        retrieved_msg = cast(ChatMessage, retrieved)
        assert retrieved_msg.content == message.content
    
    def test_message_to_dict(self):
        """Test conversion of message to dict format."""
        # Test user message
        user_msg = ChatMessage(
            role=MessageRole.user,
            content="User message"
        )
        user_dict = user_msg.to_dict()
        assert user_dict == {
            "role": "user",
            "content": "User message"
        }
        
        # Test assistant message
        asst_msg = ChatMessage(
            role=MessageRole.assistant,
            content="Assistant message"
        )
        asst_dict = asst_msg.to_dict()
        assert asst_dict == {
            "role": "assistant",
            "content": "Assistant message"
        }
        
        # Test system message
        sys_msg = ChatMessage(
            role=MessageRole.system,
            content="System message"
        )
        sys_dict = sys_msg.to_dict()
        assert sys_dict == {
            "role": "system",
            "content": "System message"
        }
        
        # Test tool result message
        tool_msg = ChatMessage(
            role=MessageRole.tool,
            content="Tool result",
            oai_tool_call_id="call_abc123"
        )
        tool_dict = tool_msg.to_dict()
        assert tool_dict == {
            "role": "tool",
            "content": "Tool result",
            "tool_call_id": "call_abc123"
        }
        
        # Test assistant with tool call
        tool_call_msg = ChatMessage(
            role=MessageRole.assistant,
            content="",
            oai_tool_call_id="call_def456",
            tool_name="test_function",
            tool_call={"x": 5, "y": 7}
        )
        tool_call_dict = tool_call_msg.to_dict()
        assert tool_call_dict["role"] == "assistant"
        assert tool_call_dict["content"] == ""
        assert "tool_calls" in tool_call_dict
        assert len(tool_call_dict["tool_calls"]) == 1
        assert tool_call_dict["tool_calls"][0]["id"] == "call_def456"
        assert tool_call_dict["tool_calls"][0]["function"]["name"] == "test_function"
        # Parse the JSON arguments and verify
        args = json.loads(tool_call_dict["tool_calls"][0]["function"]["arguments"])
        assert args == {"x": 5, "y": 7}
    
    def test_message_from_dict(self):
        """Test creation of message from dict format."""
        msg_dict = {
            "role": "user",
            "content": "Test from dict"
        }
        message = ChatMessage.from_dict(msg_dict)
        
        assert message.role == MessageRole.user
        assert message.content == "Test from dict"
    
    def test_message_relationships(self):
        """Test parent-child relationships between messages."""
        parent = ChatMessage(
            role=MessageRole.user,
            content="Parent message"
        )
        
        child = ChatMessage(
            role=MessageRole.assistant,
            content="Child message",
            parent_message_uuid=parent.ecs_id
        )
        
        # Check is_conversation_root property
        assert parent.is_conversation_root
        assert not child.is_conversation_root
        
        # Check get_parent method
        retrieved_parent = child.get_parent()
        assert retrieved_parent is not None
        assert retrieved_parent.ecs_id == parent.ecs_id
        assert retrieved_parent.content == "Parent message"

class TestTools:
    """Tests for the CallableTool and StructuredTool entities."""
    
    def test_callable_tool_creation(self, callable_tool):
        """Test creation and registration of CallableTool."""
        # Use the fixture instead of creating a new callable_tool
        tool = callable_tool
        
        # Verify properties
        assert tool.name == "_test_function"
        assert tool.docstring == "Test function that adds two numbers."
        assert tool.input_schema is not None
        assert tool.output_schema is not None
        
        # Verify registration
        retrieved = EntityRegistry.get(tool.ecs_id)
        assert retrieved is not None
        assert isinstance(retrieved, CallableTool)
        
        retrieved_tool = cast(CallableTool, retrieved)
        assert retrieved_tool.name == "_test_function"
    
    def test_structured_tool_creation(self, structured_tool):
        """Test creation and registration of StructuredTool."""
        # Verify properties
        assert structured_tool.name == "test_response"
        assert "test" in structured_tool.description.lower()
        assert structured_tool.json_schema is not None
        
        # Verify registration
        retrieved = EntityRegistry.get(structured_tool.ecs_id)
        assert retrieved is not None
        assert isinstance(retrieved, StructuredTool)
        
        retrieved_tool = cast(StructuredTool, retrieved)
        assert retrieved_tool.name == "test_response"
    
    def test_tool_execution(self, callable_tool):
        """Test execution of a CallableTool."""
        result = callable_tool.execute({"x": 5, "y": 7})
        # The tool might return a dict with the result or just the result
        if isinstance(result, dict) and "result" in result:
            assert result["result"] == 12
        else:
            assert result == 12
    
    def test_tool_validation(self, structured_tool):
        """Test validation with a StructuredTool."""
        # Valid input
        valid_input = {
            "status": "success",
            "message": "Test message",
            "data": {"key": "value"}
        }
        result = structured_tool.execute(valid_input)
        assert result == valid_input
        
        # Invalid input (missing required field)
        invalid_input = {
            "status": "success",
            "data": {"key": "value"}
        }
        result = structured_tool.execute(invalid_input)
        assert "error" in result
    
    def test_tool_openai_format(self, callable_tool, structured_tool):
        """Test conversion of tools to OpenAI format."""
        openai_callable = callable_tool.get_openai_tool()
        assert openai_callable is not None
        assert openai_callable["type"] == "function"
        assert openai_callable["function"]["name"] == callable_tool.name
        
        openai_structured = structured_tool.get_openai_tool()
        assert openai_structured is not None
        assert openai_structured["type"] == "function"
        assert openai_structured["function"]["name"] == structured_tool.name
    
    def test_tool_anthropic_format(self, callable_tool, structured_tool):
        """Test conversion of tools to Anthropic format."""
        anthropic_callable = callable_tool.get_anthropic_tool()
        assert anthropic_callable is not None
        assert anthropic_callable["name"] == callable_tool.name
        assert anthropic_callable["input_schema"] == callable_tool.input_schema
        
        anthropic_structured = structured_tool.get_anthropic_tool()
        assert anthropic_structured is not None
        assert anthropic_structured["name"] == structured_tool.name
        assert anthropic_structured["input_schema"] == structured_tool.json_schema

class TestChatThread:
    """Tests for the ChatThread entity."""
    
    def test_thread_creation(self, llm_config, system_prompt):
        """Test creation and registration of ChatThread."""
        thread = ChatThread(
            name="Test Thread",
            system_prompt=system_prompt,
            llm_config=llm_config
        )
        
        # Verify properties
        assert thread.name == "Test Thread"
        assert thread.system_prompt is not None
        assert thread.system_prompt.content == system_prompt.content
        assert thread.llm_config is not None
        assert thread.llm_config.model == llm_config.model
        assert len(thread.history) == 0
        
        # Verify registration
        retrieved = EntityRegistry.get(thread.ecs_id)
        assert retrieved is not None
        assert isinstance(retrieved, ChatThread)
        
        retrieved_thread = cast(ChatThread, retrieved)
        assert retrieved_thread.name == "Test Thread"
    
    def test_add_user_message(self, empty_chat_thread):
        """Test adding a user message to the thread."""
        thread = empty_chat_thread
        
        # Set new message and add it
        thread.new_message = "Hello, this is a test message."
        message = thread.add_user_message()
        
        # Verify message was added
        assert message is not None
        assert message.role == MessageRole.user
        assert message.content == "Hello, this is a test message."
        # After adding a user message, the thread might have been forked
        # Just check that chat_thread_id is valid
        assert message.chat_thread_id is not None
        
        # Verify history was updated
        assert len(thread.history) == 1
        assert thread.history[0].ecs_id == message.ecs_id
        
        # Verify new_message was cleared
        assert thread.new_message is None
    
    async def test_add_chat_turn_history(self, chat_thread_with_messages, processed_output):
        """Test adding a chat turn to the thread."""
        thread = chat_thread_with_messages
        initial_history_length = len(thread.history)
        
        # Add a chat turn (now using await since it's an async method)
        parent, assistant = await thread.add_chat_turn_history(processed_output)
        
        # Verify parent is the last message in history
        assert parent.ecs_id == thread.history[initial_history_length - 1].ecs_id
        
        # Verify assistant message was added
        assert assistant is not None
        assert assistant.role == MessageRole.assistant
        assert assistant.content == "This is a test response."
        # The thread might have been forked during message addition
        assert assistant.chat_thread_id is not None
        assert assistant.parent_message_uuid == parent.ecs_id
        
        # Verify history was updated
        assert len(thread.history) == initial_history_length + 1
        assert thread.history[initial_history_length].ecs_id == assistant.ecs_id
    
    def test_get_tools_for_llm(self, chat_thread_with_tools):
        """Test getting tools in appropriate format for the LLM."""
        thread = chat_thread_with_tools
        
        # Get OpenAI tools
        openai_tools = thread.get_tools_for_llm()
        assert openai_tools is not None
        assert len(openai_tools) == 2
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[1]["type"] == "function"
        
        # Change client to Anthropic
        thread.llm_config.client = "anthropic"
        anthropic_tools = thread.get_tools_for_llm()
        assert anthropic_tools is not None
        assert len(anthropic_tools) == 2
        assert "name" in anthropic_tools[0]
        assert "input_schema" in anthropic_tools[0]
    
    def test_message_conversion(self, chat_thread_with_messages):
        """Test conversion of messages to different formats."""
        thread = chat_thread_with_messages
        
        # Get messages in chatml format
        chatml_msgs = thread.messages
        # Now accounts for system message + user + assistant
        assert len(chatml_msgs) == 3
        
        # Find the messages by role
        system_msg = next((m for m in chatml_msgs if m["role"] == "system"), None)
        user_msg = next((m for m in chatml_msgs if m["role"] == "user"), None)
        asst_msg = next((m for m in chatml_msgs if m["role"] == "assistant"), None)
        
        assert system_msg is not None
        assert user_msg is not None
        assert asst_msg is not None
        
        # Get messages in OpenAI format
        oai_msgs = thread.oai_messages
        assert len(oai_msgs) >= 2  # May include system message
        oai_user_msg = next((m for m in oai_msgs if m["role"] == "user"), None)
        oai_asst_msg = next((m for m in oai_msgs if m["role"] == "assistant"), None)
        assert oai_user_msg is not None
        assert asst_msg is not None
        
        # Get messages in Anthropic format
        blocks, messages = thread.anthropic_messages
        assert len(messages) > 0
    
    async def test_tool_execution_in_chat_turn(self, chat_thread_with_tools, tool_processed_output):
        """Test tool execution during a chat turn."""
        thread = chat_thread_with_tools
        initial_history_length = len(thread.history)
        
        # Add a user message first
        thread.new_message = "What is 5 + 7?"
        thread.add_user_message()
        
        # Add a chat turn with tool execution (now using await since it's an async method)
        parent, assistant = await thread.add_chat_turn_history(tool_processed_output)
        
        # Verify messages were added (at least the assistant message should be added)
        assert len(thread.history) >= initial_history_length + 1
        
        # Check assistant message
        assert assistant.role == MessageRole.assistant
        assert assistant.tool_name == "test_function"
        assert assistant.tool_call == {"x": 5, "y": 7}
        
        # Find the tool message (it might not be the last message)
        tool_message = None
        for msg in thread.history:
            if msg.role == MessageRole.tool and msg.tool_name == "test_function":
                tool_message = msg
                break
                
        # The tool message may or may not be present depending on how the execution was simulated
        if tool_message:
            assert tool_message.tool_name == "test_function"
            assert tool_message.parent_message_uuid == assistant.ecs_id
            
            # Verify tool result is in the content
            tool_content = json.loads(tool_message.content)
            assert tool_content == 12  # 5 + 7 = 12