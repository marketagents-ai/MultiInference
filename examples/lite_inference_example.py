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
#set logging level to debug
logging.basicConfig(level=logging.DEBUG)
async def main():
    load_dotenv()
    EntityRegistry()
    CallableRegistry()
    oai_request_limits = RequestLimits(max_requests_per_minute=50000, max_tokens_per_minute=2000000000)
    lite_llm_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
    anthropic_request_limits = RequestLimits(max_requests_per_minute=50, max_tokens_per_minute=20000)
    lite_llm_model = "deephermes-3-llama-3-8b-preview-mlx"
    anthropic_model = "claude-3-5-sonnet-latest"




    orchestrator = InferenceOrchestrator(oai_request_limits=oai_request_limits, litellm_request_limits=lite_llm_request_limits, anthropic_request_limits=anthropic_request_limits)

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
            llm_config=LLMConfig(client=client, model=model, response_format=response_format,max_tokens=1000)
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
    openai_chats = create_chats(LLMClient.openai, "gpt-4o-mini",[ResponseFormat.tool],1)#+create_chats(LLMClient.openai, "gpt-4o-mini",[ResponseFormat.text],1)
    litellm_chats = create_chats(LLMClient.litellm, lite_llm_model,[ResponseFormat.tool],5)+create_chats(LLMClient.litellm, lite_llm_model,[ResponseFormat.text],5)
    anthropic_chats = create_chats(LLMClient.anthropic, anthropic_model,[ResponseFormat.tool],5)+create_chats(LLMClient.anthropic, anthropic_model,[ResponseFormat.text],5)
    


    # print(chats[0].llm_config)
    print("Running parallel completions...")
    all_chats = openai_chats+ litellm_chats+anthropic_chats
    all_chats = openai_chats
    # all_chats = [litellm_chats[0]]
    # all_chats = 
    start_time = time.time()
    # with Session(engine) as session:
    completion_results = await orchestrator.run_parallel_ai_completion(all_chats)
 


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

if __name__ == "__main__":
    all_chats = asyncio.run(main())
    print(all_chats[0].history)
    #print mermaid graph of first chat
    mermaid_str = EntityRegistry.get_lineage_mermaid(all_chats[0].lineage_id)
    print(mermaid_str)
    #strip the mermaid declaration
    mermaid_str = mermaid_str.split("```mermaid")[1]
    mermaid_str = mermaid_str.split("```")[0]
    # Write the mermaid code to a file
    mermaid_to_image(mermaid_str, "chat_thread_diff.png")

