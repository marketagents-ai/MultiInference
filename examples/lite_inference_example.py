import asyncio
from dotenv import load_dotenv
from minference.lite.inference import InferenceOrchestrator, RequestLimits
from minference.lite.models import ChatThread, LLMConfig, CallableTool, LLMClient,ResponseFormat, SystemPrompt, StructuredTool
from typing import Literal, List
from minference.lite.enregistry import EntityRegistry
from minference.lite.caregistry import CallableRegistry
import time
import os

async def main():
    load_dotenv()
    EntityRegistry()
    CallableRegistry()
    oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)





    orchestrator = InferenceOrchestrator(oai_request_limits=oai_request_limits)

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
    system_string = SystemPrompt(content="You are a helpful assistant that tells programmer jokes.", name= "joke_teller")
    # Create chats for different JSON modes and tool usage
    def create_chats(client:LLMClient, model, response_formats : List[ResponseFormat]= [ResponseFormat.text], count=1) -> List[ChatThread]:
        chats : List[ChatThread] = []
        for response_format in response_formats:
            llm_config=LLMConfig(client=client, model=model, response_format=response_format,max_tokens=250)
            for i in range(count):
                chats.append(
                    ChatThread(
                    
                    system_prompt=system_string,
                    new_message=f"Tell me a programmer joke about the number {i}.",
                    llm_config=llm_config,
                    structured_output=structured_tool,
                    
                )
            )
        return chats
    



    # OpenAI chats
    # openai_chats = create_chats("openai", "gpt-4o-mini",[ResponseFormat.text,ResponseFormat.json_beg,ResponseFormat.json_object,ResponseFormat.structured_output,ResponseFormat.tool],1)
    openai_chats = create_chats(LLMClient.openai, "gpt-4o-mini",[ResponseFormat.tool],5)
    
    
    chats_id = [chat.id for chat in openai_chats]
        

    # print(chats[0].llm_config)
    print("Running parallel completions...")
    all_chats = openai_chats
    start_time = time.time()
    # with Session(engine) as session:
    completion_results = await orchestrator.run_parallel_ai_completion(openai_chats)
    for chat in openai_chats:

        chat.new_message = "And why is it funny?"
    second_step_completion_results = await orchestrator.run_parallel_ai_completion(openai_chats)
    end_time = time.time()
    total_time = end_time - start_time

    # Print results
    num_text = 0
    num_json = 0
    total_calls = 0
    return openai_chats


if __name__ == "__main__":
    openai_chats = asyncio.run(main())
