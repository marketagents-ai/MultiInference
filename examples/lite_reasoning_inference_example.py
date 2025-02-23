import asyncio
from dotenv import load_dotenv
from minference.threads.inference import InferenceOrchestrator, RequestLimits
from minference.threads.models import ChatMessage, ChatThread, LLMConfig, CallableTool, LLMClient,ResponseFormat, SystemPrompt, StructuredTool, Usage
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
    oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)






    orchestrator = InferenceOrchestrator(oai_request_limits=oai_request_limits)
    EntityRegistry.set_inference_orchestrator(orchestrator)
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
            llm_config=LLMConfig(client=client, model=model, response_format=response_format,max_tokens=1000,reasoner=True)
            for i in range(count):
                chats.append(
                    ChatThread(
                    
                    system_prompt=system_string,
                    new_message=f"Tell me a programmer joke about the number {i}.",
                    llm_config=llm_config,
                    forced_output=structured_tool,
                    
                )
            )
        return chats
    



    # OpenAI chats
    # openai_chats = create_chats("openai", "gpt-4o-mini",[ResponseFormat.text,ResponseFormat.json_beg,ResponseFormat.json_object,ResponseFormat.structured_output,ResponseFormat.tool],1)
    # openai_chats = create_chats(LLMClient.openai, "o1-2024-12-17",[ResponseFormat.text],1)
    openai_chats = create_chats(LLMClient.openai, "o1-2024-12-17",[ResponseFormat.tool],1)



    # print(chats[0].llm_config)
    print("Running parallel completions...")
    all_chats = openai_chats
    all_chats = all_chats
    start_time = time.time()
    # with Session(engine) as session:
    completion_results = await orchestrator.run_parallel_ai_completion(all_chats)
    all_messages = EntityRegistry.list_by_type(ChatMessage)

    print("messages object",all_messages)



    for chat in all_chats:


        chat.new_message = "And why is it funny?"
    second_step_completion_results = await orchestrator.run_parallel_ai_completion(all_chats)
    end_time = time.time()
    total_time = end_time - start_time


    # Print results
    num_text = 0
    num_json = 0
    total_calls = 0
    return openai_chats


if __name__ == "__main__":
    all_chats = asyncio.run(main())
    print(EntityRegistry.list_by_type(Usage))
    for chat in all_chats:
        print(chat.history[-1])
