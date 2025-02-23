"""
Example usage of sql_models.py illustrating:
 - UUID-based primary keys
 - parent_id / lineage_id for versioning
 - bridging tables for many-to-many relationships
 - basic from_entity() / to_entity() methods
"""

from typing import List, Optional, Dict, Any, Union
from uuid import UUID, uuid4
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship, Session, create_engine

from minference.threads.models import (
    ChatThread, ChatMessage, LLMConfig, Usage, GeneratedJsonObject,
    CallableTool, StructuredTool, RawOutput, ProcessedOutput, SystemPrompt,
    MessageRole, LLMClient, ToolType, ResponseFormat
)

from minference.threads.sql_models import (
    ThreadMessageLinkSQL, ThreadToolLinkSQL, ChatMessageSQL, UsageSQL,
    ToolSQL, SystemPromptSQL, ChatThreadSQL, LLMConfigSQL,
    message_role_to_literal, literal_to_message_role
)
from minference.ecs.entity import EntityRegistry

###############################################################################
# Example Usage
###############################################################################

def create_example_chat_thread() -> ChatThread:
    """Create an example ChatThread with messages and tools"""
    system_prompt = SystemPrompt(
        id=uuid4(),
        lineage_id=uuid4(),
        name="Example System Prompt",
        content="You are a helpful assistant."
    )

    llm_config = LLMConfig(
        id=uuid4(),
        client=LLMClient.openai,
        model="gpt-4",
        max_tokens=400,
        temperature=0,
        response_format=ResponseFormat.text
    )

    tool = StructuredTool(
        id=uuid4(),
        lineage_id=uuid4(),
        name="example_tool",
        description="An example tool",
        instruction_string="Follow this schema:",
        json_schema={"type": "object", "properties": {"key": {"type": "string"}}}
    )

    message = ChatMessage(
        id=uuid4(),
        lineage_id=uuid4(),
        role=MessageRole.user,
        content="Hello!",
        timestamp=datetime.utcnow()
    )

    thread = ChatThread(
        id=uuid4(),
        lineage_id=uuid4(),
        name="Example Thread",
        system_prompt=system_prompt,
        history=[message],
        llm_config=llm_config,
        tools=[tool]
    )

    return thread

def example_storage_operations(engine) -> None:
    """Example of storing and retrieving a ChatThread"""
    # Create an example thread
    thread = create_example_chat_thread()

    # Store it in the database
    with Session(engine) as session:
        # Convert to SQL models
        if thread.system_prompt:
            system_prompt_sql = SystemPromptSQL.from_entity(thread.system_prompt)
            session.add(system_prompt_sql)

        if thread.llm_config:
            llm_config_sql = LLMConfigSQL.from_entity(thread.llm_config)
            session.add(llm_config_sql)

        thread_sql = ChatThreadSQL.from_entity(thread)
        session.add(thread_sql)
        
        # Add tools
        tool_sql_list = [ToolSQL.from_entity(tool) for tool in thread.tools]
        for tool_sql in tool_sql_list:
            session.add(tool_sql)
        
        # Add messages
        message_sql_list = [ChatMessageSQL.from_entity(msg) for msg in thread.history]
        for message_sql in message_sql_list:
            session.add(message_sql)
            
        # Link messages and tools to thread
        thread_sql.messages.extend(message_sql_list)
        thread_sql.tools.extend(tool_sql_list)
        
        session.commit()

        # Retrieve and verify
        loaded_thread_sql = session.get(ChatThreadSQL, thread.id)
        if loaded_thread_sql is None:
            raise ValueError(f"Failed to retrieve thread with id {thread.id}")
            
        loaded_thread = loaded_thread_sql.to_entity()
        
        print(f"Stored and retrieved thread: {loaded_thread.name}")
        print(f"Number of messages: {len(loaded_thread.history)}")
        print(f"Number of tools: {len(loaded_thread.tools)}")

if __name__ == "__main__":
    EntityRegistry()
    # Create tables
    engine = create_engine("sqlite:///example.db")
    SQLModel.metadata.create_all(engine)
    
    # Run example
    example_storage_operations(engine)
