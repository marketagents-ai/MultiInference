import asyncio
from dotenv import load_dotenv
from minference.threads.inference import InferenceOrchestrator, RequestLimits
from minference.threads.models import ChatMessage, ChatThread, LLMConfig, CallableTool, LLMClient, ResponseFormat, SystemPrompt, StructuredTool, Usage
from typing import Literal, List
from minference.ecs.entity import EntityRegistry
from minference.ecs.caregistry import CallableRegistry
import time
from minference.clients.utils import msg_dict_to_oai, msg_dict_to_anthropic, parse_json_string

import os

async def main():
    load_dotenv()
    EntityRegistry()
    CallableRegistry()
    
    # Only configure OpenRouter limits
    openrouter_request_limits = RequestLimits(
        max_requests_per_minute=50, 
        max_tokens_per_minute=40000,
        provider="openrouter"
    )

    # Initialize orchestrator with only OpenRouter configuration
    orchestrator = InferenceOrchestrator(
        openrouter_request_limits=openrouter_request_limits
    )

    # Define the JSON schema for the structured tool
    json_schema = {
        "type": "object",
        "properties": {
            "joke": {"type": "string"},
            "explanation": {"type": "string"}
        },
        "required": ["joke", "explanation"],
        "additionalProperties": False
    }

    structured_tool = StructuredTool(
        json_schema=json_schema,
        name="tell_joke",
        description="Generate a programmer joke with explanation"
    )
    
    system_string = SystemPrompt(
        content="You are a helpful assistant that tells programmer jokes.", 
        name="joke_teller"
    )

    def create_chats(response_formats=[ResponseFormat.text], count=1) -> List[ChatThread]:
        chats: List[ChatThread] = []
        for response_format in response_formats:
            llm_config = LLMConfig(
                client=LLMClient.openrouter,
                model="deepseek/deepseek-r1",  # Example OpenRouter model
                response_format=response_format,
                max_tokens=5000
            )
            for i in range(count):
                chats.append(
                    ChatThread(
                        
                        new_message=f"Find the nearest prime number to 23+{i} times {i}.",
                        llm_config=llm_config,
                        forced_output=structured_tool if response_format == ResponseFormat.tool else None,
                    )
                )
        return chats

    # Create chats for both text and tool formats
    all_chats = create_chats([ResponseFormat.tool], 5) + create_chats([ResponseFormat.text], 5)
    all_chats = create_chats([ResponseFormat.text], 1)
    print("Running parallel completions...")
    start_time = time.time()
    
    # Run first round of completions
    completion_results = await orchestrator.run_parallel_ai_completion(all_chats)
    
    # Add follow-up question to all chats
    for chat in all_chats:
        chat.new_message = "And why is it funny?"
    
    # Run second round of completions
    second_step_completion_results = await orchestrator.run_parallel_ai_completion(all_chats)
    
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total time taken: {total_time:.2f} seconds")
    return all_chats

if __name__ == "__main__":
    all_chats = asyncio.run(main())
    # Print usage statistics
    print("\nUsage Statistics:")
    for usage in EntityRegistry.list_by_type(Usage):
        print(usage)
    for chat in all_chats:
        print(chat.history[-1])
