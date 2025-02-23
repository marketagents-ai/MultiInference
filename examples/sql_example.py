"""
Example usage of sql_models.py illustrating:
 - Error handling and assertions
 - One-to-one relationships (Message-Usage)
 - Many-to-many relationships (Thread-Tools)
 - Entity conversion and validation
 - Transaction management
 - Tool registration and validation
"""

from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime, UTC
from sqlmodel import SQLModel, Session, create_engine
import pytest
from minference.ecs.entity import EntityRegistry
from minference.ecs.caregistry import CallableRegistry
from minference.threads.models import (
    ChatThread, ChatMessage, LLMConfig, Usage, GeneratedJsonObject,
    CallableTool, StructuredTool, SystemPrompt, MessageRole, LLMClient,
    ToolType, ResponseFormat
)

from minference.threads.sql_models import (
    ChatThreadSQL, ChatMessageSQL, LLMConfigSQL, UsageSQL,
    GeneratedJsonObjectSQL, ToolSQL, SystemPromptSQL
)

def create_example_entities():
    """Create example entities for testing"""
    # Create a system prompt
    system_prompt = SystemPrompt(
        id=uuid4(),
        lineage_id=uuid4(),
        name="Example System Prompt",
        content="You are a helpful assistant."
    )

    # Create an LLM config with structured output
    llm_config = LLMConfig(
        id=uuid4(),
        client=LLMClient.openai,
        model="gpt-4",
        max_tokens=400,
        temperature=0,
        response_format=ResponseFormat.json_object
    )

    # Create a structured tool
    tool = StructuredTool(
        id=uuid4(),
        lineage_id=uuid4(),
        name="calculate_price",
        description="Calculate total price including tax",
        json_schema={
            "type": "object",
            "properties": {
                "price": {"type": "number", "description": "Base price"},
                "tax_rate": {"type": "number", "description": "Tax rate as decimal"}
            },
            "required": ["price", "tax_rate"]
        }
    )

    # Create a callable tool
    callable_tool = CallableTool(
        id=uuid4(),
        lineage_id=uuid4(),
        name="multiply",
        docstring="Multiply two numbers",
        callable_text="def multiply(x: float, y: float) -> float:\n    return x * y"
    )

    # Create a message with usage
    usage = Usage(
        id=uuid4(),
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        model="gpt-4"
    )

    # Create a user message with UTC time
    user_message = ChatMessage(
        id=uuid4(),
        lineage_id=uuid4(),
        role=MessageRole.user,
        content="Calculate price for $100 with 8.5% tax",
        timestamp=datetime.now(UTC)
    )

    # Create an assistant message with UTC time
    assistant_message = ChatMessage(
        id=uuid4(),
        lineage_id=uuid4(),
        role=MessageRole.assistant,
        content="",
        timestamp=datetime.now(UTC),
        usage=usage,
        tool_name="calculate_price",
        tool_call={"price": 100, "tax_rate": 0.085}
    )

    # Create a thread with tool usage
    thread = ChatThread(
        id=uuid4(),
        lineage_id=uuid4(),
        name="Price Calculation Thread",
        system_prompt=system_prompt,
        history=[user_message, assistant_message],
        llm_config=llm_config,
        tools=[tool, callable_tool]
    )

    return thread, user_message, assistant_message, usage, tool, callable_tool, llm_config, system_prompt

def test_sql_models(engine):
    """Test SQL models with error handling and assertions"""
    thread, user_msg, assistant_msg, usage, tool, callable_tool, llm_config, system_prompt = create_example_entities()

    try:
        with Session(engine) as session:
            # Convert entities to SQL models
            try:
                system_prompt_sql = SystemPromptSQL.from_entity(system_prompt)
                llm_config_sql = LLMConfigSQL.from_entity(llm_config)
                tool_sql = ToolSQL.from_entity(tool)
                callable_tool_sql = ToolSQL.from_entity(callable_tool)
                usage_sql = UsageSQL.from_entity(usage)
                user_msg_sql = ChatMessageSQL.from_entity(user_msg)
                assistant_msg_sql = ChatMessageSQL.from_entity(assistant_msg)
                thread_sql = ChatThreadSQL.from_entity(thread)
            except Exception as e:
                raise ValueError(f"Failed to convert entities to SQL models: {str(e)}")

            # Add models to session in correct order
            try:
                # First add independent models
                session.add(system_prompt_sql)
                session.add(llm_config_sql)
                session.add(tool_sql)
                session.add(callable_tool_sql)
                session.add(usage_sql)
                
                # Then add messages with relationships
                user_msg_sql.usage = None  # User message doesn't have usage
                assistant_msg_sql.usage = usage_sql
                session.add(user_msg_sql)
                session.add(assistant_msg_sql)
                
                # Finally add thread with its relationships
                thread_sql.messages = [user_msg_sql, assistant_msg_sql]
                thread_sql.tools = [tool_sql, callable_tool_sql]
                session.add(thread_sql)
                
            except Exception as e:
                raise ValueError(f"Failed to add models to session: {str(e)}")

            # Commit changes
            try:
                session.commit()
            except Exception as e:
                session.rollback()
                raise ValueError(f"Failed to commit changes: {str(e)}")

            # Verify storage and relationships
            try:
                # Retrieve thread
                loaded_thread_sql = session.get(ChatThreadSQL, thread.id)
                assert loaded_thread_sql is not None, "Failed to retrieve thread"
                
                # Convert back to entity
                loaded_thread = loaded_thread_sql.to_entity()
                
                # Verify thread attributes
                assert loaded_thread.name == thread.name, "Thread name mismatch"
                assert len(loaded_thread.history) == 2, "Message count mismatch"
                assert len(loaded_thread.tools) == 2, "Tool count mismatch"
                
                # Verify tool attributes
                loaded_tool = next(t for t in loaded_thread.tools if isinstance(t, StructuredTool))
                assert loaded_tool.name == tool.name, "Tool name mismatch"
                assert loaded_tool.json_schema == tool.json_schema, "Tool schema mismatch"

                loaded_callable = next(t for t in loaded_thread.tools if isinstance(t, CallableTool))
                assert loaded_callable.name == callable_tool.name, "Callable tool name mismatch"
                assert loaded_callable.callable_text == callable_tool.callable_text, "Callable text mismatch"
                
                # Verify message and usage
                loaded_assistant_msg = next(m for m in loaded_thread.history if m.role == MessageRole.assistant)
                assert loaded_assistant_msg.tool_name == assistant_msg.tool_name, "Tool name mismatch"
                assert loaded_assistant_msg.tool_call == assistant_msg.tool_call, "Tool call mismatch"
                assert loaded_assistant_msg.usage is not None, "Missing usage"
                assert loaded_assistant_msg.usage.total_tokens == usage.total_tokens, "Usage tokens mismatch"
                
                print("✅ All assertions passed!")
                
            except AssertionError as e:
                raise ValueError(f"Verification failed: {str(e)}")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise

def main():
    """Main function to run the example"""
    EntityRegistry()
    CallableRegistry()
    # Create tables
    engine = create_engine("sqlite:///example.db")
    SQLModel.metadata.create_all(engine)
    
    # Run test
    test_sql_models(engine)

if __name__ == "__main__":
    main()
