import sys
from minference.lite.models import (
    ChatThread, ChatMessage, SystemPrompt, LLMConfig,
    LLMClient, ResponseFormat, MessageRole, Usage,
    GeneratedJsonObject, RawOutput, ProcessedOutput
)
from minference.entity import EntityRegistry
from minference.lite.inference import process_outputs_and_execute_tools, run_parallel_ai_completion, InferenceOrchestrator
from uuid import uuid4
import asyncio
from datetime import datetime
import logging
import time

# Configure logging

# Initialize registry and make it globally available
EntityRegistry()
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Add this line to ensure we capture all loggers
logging.getLogger().setLevel(logging.DEBUG)
# Clear any existing state
EntityRegistry.clear()
EntityRegistry.clear_logs()

def print_thread_state(chat_thread: ChatThread, label: str):
    """Helper to print thread state"""
    print(f"\n=== {label} ===")
    print(f"Thread ID: {chat_thread.id}")
    print(f"Thread Lineage ID: {chat_thread.lineage_id}")
    print(f"LLMConfig type: {type(chat_thread.llm_config)}")
    print(f"LLMConfig ID: {chat_thread.llm_config.id}")
    print(f"LLMConfig client: {chat_thread.llm_config.client}")
    print(f"LLMConfig model: {chat_thread.llm_config.model}")
    print(f"Message Count: {len(chat_thread.history)}")
    if chat_thread.history:
        print(f"Last Message type: {type(chat_thread.history[-1])}")

async def main():
    # Create orchestrator
    orchestrator = InferenceOrchestrator()
    
    # Create initial config
    llm_config = LLMConfig(
        client=LLMClient.openai,
        model="gpt-4o-mini",
        response_format=ResponseFormat.text
    )
    
    # Create initial thread
    chat_thread = ChatThread(
        name="test_chat",
        llm_config=llm_config,
        new_message="This is a test message"
    )
    
    print_thread_state(chat_thread, "Initial Thread State")
    
    # Run first completion
    print("\n=== Running First Completion ===")
    completion_outputs = await run_parallel_ai_completion([chat_thread], orchestrator)
    
    # Get the updated thread from the registry
    updated_thread = EntityRegistry.get(chat_thread.id)
    if not updated_thread:
        print("Error: Could not find updated thread in registry")
        return
        
    print_thread_state(updated_thread, "After First Completion")
    
    # Run second completion
    print("\n=== Running Second Completion ===")
    updated_thread.new_message = "Another test message"
    completion_outputs = await run_parallel_ai_completion([updated_thread], orchestrator)
    
    # Get the final thread state
    final_thread_id = updated_thread.id if updated_thread else chat_thread.id
    final_thread = EntityRegistry.get(final_thread_id)
    if final_thread:
        print_thread_state(final_thread, "After Second Completion")
    else:
        print("Error: Could not find final thread in registry")
    
    # Print final registry status
    print("\n=== Final Registry Status ===")
    status = EntityRegistry.get_registry_status()
    print(f"Total entities: {status['total_items']}")
    print(f"Total lineages: {status['total_lineages']}")
    print(f"Total versions: {status['total_versions']}")
    
    # Print lineage tree
    print("\n=== Lineage Tree ===")
    print(EntityRegistry.get_lineage_mermaid(chat_thread.lineage_id))

if __name__ == "__main__":
    asyncio.run(main()) 