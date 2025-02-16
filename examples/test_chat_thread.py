import sys
from minference.lite.models import (
    ChatThread, ChatMessage, SystemPrompt, LLMConfig,
    LLMClient, ResponseFormat, MessageRole, Usage,
    GeneratedJsonObject
)
from minference.entity import EntityRegistry
from uuid import uuid4
import asyncio
from datetime import datetime
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize registry and make it globally available
EntityRegistry()

# Clear any existing state
EntityRegistry.clear()
EntityRegistry.clear_logs()

def create_mock_usage() -> Usage:
    """Create a mock usage object for testing"""
    return Usage(
        model="gpt-4",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150
    )

def create_mock_json_object() -> GeneratedJsonObject:
    """Create a mock JSON object for testing"""
    return GeneratedJsonObject(
        name="test_tool",
        object={"key": "value"},
        tool_call_id=str(uuid4())
    )

async def main():
    # Create a basic system prompt
    system_prompt = SystemPrompt(
        name="test_system",
        content="You are a test assistant."
    )

    # Create a chat thread
    chat_thread = ChatThread(
        name="test_chat",
        system_prompt=system_prompt,
        llm_config=LLMConfig(
            client=LLMClient.openai,
            model="gpt-4",
            response_format=ResponseFormat.text
        )
    )

    print("\n=== Initial Chat Thread State ===")
    print(f"Thread ID: {chat_thread.id}")
    print(f"Thread Lineage ID: {chat_thread.lineage_id}")
    print(f"Message Count: {len(chat_thread.history)}")

    # Add a user message
    chat_thread.new_message = "Hello, this is a test message"
    user_message = ChatMessage(
        role=MessageRole.user,
        content=chat_thread.new_message,
        chat_thread_uuid=chat_thread.id
    )
    chat_thread.history.append(user_message)
    chat_thread.new_message = None

    print("\n=== After Adding User Message ===")
    print(f"Message Count: {len(chat_thread.history)}")
    print(f"Last Message Role: {chat_thread.history[-1].role}")
    print(f"Last Message Content: {chat_thread.history[-1].content}")

    # Create a mock assistant message with tool usage
    mock_json = create_mock_json_object()
    mock_usage = create_mock_usage()
    
    assistant_message = ChatMessage(
        role=MessageRole.assistant,
        content="I'm using a test tool",
        chat_thread_uuid=chat_thread.id,
        parent_message_uuid=user_message.id,
        tool_call=mock_json.object,
        tool_name=mock_json.name,
        oai_tool_call_id=mock_json.tool_call_id,
        usage=mock_usage
    )

    # Add the assistant message to history
    chat_thread.history.append(assistant_message)

    print("\n=== After Adding Assistant Message ===")
    print(f"Message Count: {len(chat_thread.history)}")
    print(f"Last Message Role: {chat_thread.history[-1].role}")
    print(f"Last Message Tool Name: {chat_thread.history[-1].tool_name}")
    print(f"Last Message Tool Call: {chat_thread.history[-1].tool_call}")

    # Create a mock tool response message
    tool_message = ChatMessage(
        role=MessageRole.tool,
        content='{"result": "test_result"}',
        chat_thread_uuid=chat_thread.id,
        parent_message_uuid=assistant_message.id,
        tool_name=mock_json.name,
        oai_tool_call_id=mock_json.tool_call_id
    )

    # Add the tool message to history
    chat_thread.history.append(tool_message)

    print("\n=== After Adding Tool Message ===")
    print(f"Message Count: {len(chat_thread.history)}")
    print(f"Last Message Role: {chat_thread.history[-1].role}")
    print(f"Last Message Content: {chat_thread.history[-1].content}")

    print("\n=== Final Registry Status ===")
    status = EntityRegistry.get_registry_status()
    print(f"Total entities: {status['total_items']}")
    print(f"Total lineages: {status['total_lineages']}")
    print(f"Total versions: {status['total_versions']}")
    print("reregistring")
    EntityRegistry.register(chat_thread)
    
    print("\n=== Lineage Tree ===")
    print(EntityRegistry.get_lineage_tree_sorted(chat_thread.lineage_id))

if __name__ == "__main__":
    asyncio.run(main()) 