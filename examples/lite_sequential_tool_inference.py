import asyncio
from dotenv import load_dotenv
from minference.threads.inference import InferenceOrchestrator, RequestLimits
from minference.threads.models import ChatMessage, ChatThread, LLMConfig, CallableTool, LLMClient,ResponseFormat, SystemPrompt, StructuredTool, Usage
from typing import Literal, List
from minference.ecs.caregistry import CallableRegistry
import time
from minference.clients.utils import msg_dict_to_oai, msg_dict_to_anthropic, parse_json_string
from minference.ecs.entity import EntityRegistry
import os
import logging
from pydantic import BaseModel
import statistics


# Example BaseModel for inputs/outputs
class NumbersInput(BaseModel):
    numbers: List[float]
    round_to: int = 2

class Stats(BaseModel):
    mean: float
    std: float

class FilterInput(BaseModel):
    numbers: List[float]
    threshold: float

class FilterOutput(BaseModel):
    filtered_numbers: List[float]
    count_removed: int

class SortInput(BaseModel):
    numbers: List[float]
    ascending: bool = True

class SortOutput(BaseModel):
    sorted_numbers: List[float]

class GoalInput(BaseModel):
    description: str
    final_result: List[float]
    initial_numbers: List[float]

class GoalOutput(BaseModel):
    goal_achieved: bool
    explanation: str

# Tool functions
def calculate_stats(input_data: NumbersInput) -> Stats:
    """Calculate mean and standard deviation of numbers."""
    return Stats(
        mean=round(statistics.mean(input_data.numbers), input_data.round_to),
        std=round(statistics.stdev(input_data.numbers), input_data.round_to)
    )

def filter_numbers(input_data: FilterInput) -> FilterOutput:
    """Filter numbers above a threshold."""
    filtered = [n for n in input_data.numbers if n <= input_data.threshold]
    return FilterOutput(
        filtered_numbers=filtered,
        count_removed=len(input_data.numbers) - len(filtered)
    )

def sort_numbers(input_data: SortInput) -> SortOutput:
    """Sort numbers in ascending or descending order."""
    return SortOutput(
        sorted_numbers=sorted(input_data.numbers, reverse=not input_data.ascending)
    )

def check_goal_achieved(input_data: GoalInput) -> GoalOutput:
    """Verify if the goal was achieved and explain the results."""
    return GoalOutput(
        goal_achieved=True,
        explanation=f"Successfully processed the numbers. Started with {input_data.initial_numbers} and ended with {input_data.final_result}"
    )

async def run_sequential_steps(orchestrator: InferenceOrchestrator, initial_chat: ChatThread, user_feedback: bool = False) -> None:
    """Run sequential steps using the orchestrator."""
    max_steps = 10
    step = 0
    chat = initial_chat
    
    while step < max_steps:
        print(f"\nExecuting step {step + 1}...")
        
        completion_results = await orchestrator.run_parallel_ai_completion([chat])
        
        if not completion_results:
            print("No completion results received")
            break
            
        result = completion_results[0]
        print(f"\nStep {step + 1} Result:")
        print(f"Content: {result.content}")
        if result.json_object:
            print(f"Tool Name: {result.json_object.name}")
            print(f"Tool Call: {result.json_object.object}")
            
        if result.json_object and result.json_object.name == "check_goal_achieved":
            print("\nGoal achieved, stopping sequence.")
            break
            
        step += 1
        
        # Set up next step if needed
        if user_feedback:
            if step < max_steps:
                chat.new_message = "Continue with the next step. and explain your rationale for choosing the next step."

async def run_parallel_chats(orchestrator, all_chats):
    tasks = [run_sequential_steps(orchestrator, chat, user_feedback=False) for chat in all_chats]
    await asyncio.gather(*tasks)

async def main():
    load_dotenv()
    
    # Initialize registries
    EntityRegistry()
    CallableRegistry()

    # Create tools
    tools : List[CallableTool | StructuredTool] = [
        CallableTool.from_callable(calculate_stats),
        CallableTool.from_callable(filter_numbers),
        CallableTool.from_callable(sort_numbers),
        CallableTool.from_callable(check_goal_achieved)
    ]

    # Example data
    example_numbers = [15, 7, 32, 9, 21, 6, 18, 25, 13, 28]
    
    system_prompt = SystemPrompt(
        content="""You are a helpful assistant that processes numerical data using available tools. 
        You have access to these tools:
        - calculate_stats: Calculate mean and standard deviation
        - filter_numbers: Filter numbers above a threshold
        - sort_numbers: Sort numbers in ascending/descending order
        - check_goal_achieved: Verify and explain the achieved results
        
        Please break down tasks into appropriate steps and use tools sequentially to achieve the goals.
        After completing all necessary calculations, use check_goal_achieved to summarize the results.
        Explain your reasoning at each step with normal text. I expect at each response both text content and tool calls. 
        The process will not stop until you use the check_goal_achieved tool. be careful not to get stuck in a loop.""",
        name="sequential_tools_system"
    )

    # Initialize orchestrator
    lite_llm_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
    lite_llm_model = "openai/NousResearch/Hermes-3-Llama-3.1-8B"
    anthropic_request_limits = RequestLimits(max_requests_per_minute=50, max_tokens_per_minute=20000)
    anthropic_model = "claude-3-5-sonnet-latest"
    orchestrator = InferenceOrchestrator(
        oai_request_limits=RequestLimits(
            max_requests_per_minute=500,

            max_tokens_per_minute=200000
        ),
        litellm_request_limits=lite_llm_request_limits,
        anthropic_request_limits=anthropic_request_limits
    )


    # Create initial chat
    oai_chat = ChatThread(
        system_prompt=system_prompt,
        new_message=f"Using the numbers {example_numbers}, please filter out numbers above 20, then sort the remaining numbers in ascending order, and calculate their statistics.",
        llm_config=LLMConfig(
            client=LLMClient.openai,
            model="gpt-4o-mini",
            response_format=ResponseFormat.auto_tools,
            max_tokens=500
        ),
        tools=tools
    )
    litellm_chat = ChatThread(
        system_prompt=system_prompt,
        new_message=f"Using the numbers {example_numbers}, please filter out numbers above 20, then sort the remaining numbers in ascending order, and calculate their statistics.",
        llm_config=LLMConfig(
            client=LLMClient.litellm,
            model=lite_llm_model,
            response_format=ResponseFormat.auto_tools,
            max_tokens=500
        ),
        tools=tools

    )
    anthropic_chat = ChatThread(
        system_prompt=system_prompt,
        new_message=f"Using the numbers {example_numbers}, please filter out numbers above 20, then sort the remaining numbers in ascending order, and calculate their statistics.",
        llm_config=LLMConfig(
            client=LLMClient.anthropic,
            model=anthropic_model,
            response_format=ResponseFormat.auto_tools,
            max_tokens=500
        ),
        tools=tools
    )

    all_chats = [oai_chat, litellm_chat, anthropic_chat]
    all_chats = [oai_chat]
    print("Starting sequential tool inference...")
    await run_parallel_chats(orchestrator, all_chats)

import subprocess
import tempfile
import os

def mermaid_to_image(mermaid_str, output_path):
    """
    Convert a Mermaid string to an image using mermaid-cli.

    :param mermaid_str: The Mermaid diagram text (string)
    :param output_path: Path (with extension .png or .svg) to write the resulting image
    """
    # Create a temporary file for the Mermaid code
    with tempfile.NamedTemporaryFile(suffix=".mmd", delete=False) as tmp:
        tmp.write(mermaid_str.encode("utf-8"))
        tmp_name = tmp.name

    # Run mermaid-cli to convert the .mmd file to an image
    subprocess.run([
        "mmdc",
        "-i", tmp_name,
        "-o", output_path
    ], check=True)

    # Clean up the temporary .mmd file
    os.remove(tmp_name)


    # await run_sequential_steps(orchestrator, chat2, user_feedback=True)
        
  

if __name__ == "__main__":
    asyncio.run(main())
    print(EntityRegistry.list_by_type(ChatMessage))
    threads = EntityRegistry.list_by_type(ChatThread)
    mermaid_str = EntityRegistry.get_lineage_mermaid(threads[0].lineage_id)
    #strip the mermaid declaration
    print(mermaid_str)
    mermaid_str = mermaid_str.split("```mermaid")[1]
    mermaid_str = mermaid_str.split("```")[0]
    
    mermaid_to_image(mermaid_str, "chat_thread_auto_tools_diff.png")


