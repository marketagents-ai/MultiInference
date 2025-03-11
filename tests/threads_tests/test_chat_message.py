"""
Tests for the ChatMessage entity in the threads module.
"""
import pytest
import json
from typing import cast, Optional
from uuid import UUID

from minference.ecs.enregistry import EntityRegistry
from minference.threads.models import ChatMessage, MessageRole

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
        retrieved = ChatMessage.get(message.ecs_id)
        assert retrieved is not None
        assert isinstance(retrieved, ChatMessage)
        assert retrieved.content == message.content
    
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
    
    def test_tool_related_messages(self):
        """Test tool-related message formats."""
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
    
    def test_message_parent_child_relationship(self):
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
        assert parent.is_conversation_root is True  # Use 'is' for boolean comparisons
        assert child.is_conversation_root is False
        
        # Check get_parent method
        retrieved_parent = child.get_parent()
        assert retrieved_parent is not None
        assert retrieved_parent.ecs_id == parent.ecs_id
        assert retrieved_parent.content == "Parent message"