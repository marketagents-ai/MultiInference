import asyncio
from dotenv import load_dotenv
from minference.lite.inference import InferenceOrchestrator, RequestLimits
from minference.lite.models import ChatThread, LLMConfig, CallableTool, LLMClient, ResponseFormat, SystemPrompt
from typing import List
from pydantic import BaseModel
import time
from minference.enregistry import EntityRegistry
from minference.caregistry import CallableRegistry

async def main():
    load_dotenv()
    # Initialize registries
    EntityRegistry()
    CallableRegistry()
    
    oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
    lite_llm_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
    lite_llm_model = "openai/NousResearch/Hermes-3-Llama-3.1-8B"
    anthropic_request_limits = RequestLimits(max_requests_per_minute=50, max_tokens_per_minute=20000)
    anthropic_model = "claude-3-5-sonnet-latest"

    orchestrator = InferenceOrchestrator(oai_request_limits=oai_request_limits, litellm_request_limits=lite_llm_request_limits, anthropic_request_limits=anthropic_request_limits)

    # Example BaseModel for inputs
    class NumbersInput(BaseModel):
        numbers: List[float]
        round_to: int = 2

    # Example BaseModel for outputs
    class Stats(BaseModel):
        mean: float
        std: float  

    # Example functions with different input/output types
    def analyze_numbers_basemodel(input_data: NumbersInput) -> Stats:
        """Calculate statistical measures using BaseModel input and output."""
        import statistics
        return Stats(
            mean=round(statistics.mean(input_data.numbers), input_data.round_to),
            std=round(statistics.stdev(input_data.numbers), input_data.round_to)
        )
    
    class MinimumAnalysis(BaseModel):
        minimum: float
    
    def analyze_minimum_basemodel(input_data: NumbersInput) -> MinimumAnalysis:
        return MinimumAnalysis(minimum=min(input_data.numbers))
    
    # Create CallableTools from functions
    analyze_number = CallableTool.from_callable(analyze_numbers_basemodel)
    analyze_minimum = CallableTool.from_callable(analyze_minimum_basemodel)
    
    example_series = [3,7,2,1,4,5,6,8]
    system_string = SystemPrompt(
        content="You are a helpful assistant with access to multiple tools, use tools whenever possible. Include a text explanation of your reasoning in using the tools.",
        name="example_system_string"
    )
    
    # Create chats for different response formats
    def create_chats(client: LLMClient, model: str, response_formats: List[ResponseFormat] = [ResponseFormat.text], count=1) -> List[ChatThread]:
        chats: List[ChatThread] = []
        for response_format in response_formats:
            llm_config = LLMConfig(
                client=client,
                model=model,
                response_format=response_format,
                max_tokens=250
            )
            for i in range(count):
                chats.append(
                    ChatThread(
                        system_prompt=system_string,
                        new_message=f"Calculate the mean and standard deviation of the following numbers: {[x*(i+1) for x in example_series]}",
                        llm_config=llm_config,
                        tools=[analyze_number, analyze_minimum]
                    )
                )
        return chats

    # Create OpenAI chats with auto tools format
    openai_chats = create_chats(LLMClient.openai, "gpt-4o-mini", [ResponseFormat.auto_tools], 1)
    litellm_chats = create_chats(LLMClient.litellm, lite_llm_model, [ResponseFormat.auto_tools], 1)
    anthropic_chats = create_chats(LLMClient.anthropic, anthropic_model, [ResponseFormat.auto_tools], 1)
    all_chats = openai_chats + litellm_chats + anthropic_chats
    chats_id = [chat.id for chat in all_chats]


    print("Running parallel completions...")
    start_time = time.time()
    
    # First completion
    completion_results = await orchestrator.run_parallel_ai_completion(all_chats)
    for result in completion_results:
        print(f"Printing result: {result.json_object}")
    

    # Update messages and run second completion
    for chat in all_chats:
        chat.new_message = "and which one is the smallest?"
    second_step_completion_results = await orchestrator.run_parallel_ai_completion(all_chats)

    
    # Update messages and run third completion
    for chat in all_chats:
        chat.new_message = "and which one is the biggest?"
    third_step_completion_results = await orchestrator.run_parallel_ai_completion(all_chats)

    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time:.2f} seconds")
    return all_chats
if __name__ == "__main__":
    openai_chats = asyncio.run(main())
    print(openai_chats[0].history)
    print(openai_chats[0].get_all_usages())