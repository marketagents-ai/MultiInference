import asyncio
import logging
import os
import time

from dotenv import load_dotenv
from typing import Literal, List, Dict, Type, cast

# SQLModel / database imports
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Imports from your project
from minference.threads.inference import InferenceOrchestrator, RequestLimits
from minference.threads.models import (
    ChatMessage, ChatThread, LLMConfig, CallableTool,
    LLMClient, ResponseFormat, SystemPrompt, StructuredTool, Usage,
    GeneratedJsonObject, RawOutput, ProcessedOutput
)
from minference.threads.sql_models import (
    ChatThreadSQL, ChatMessageSQL, LLMConfigSQL, UsageSQL,
    ToolSQL, SystemPromptSQL, GeneratedJsonObjectSQL, RawOutputSQL, ProcessedOutputSQL,
    ENTITY_MODEL_MAP, Base  # Import Base from sql_models
)
from minference.ecs.caregistry import CallableRegistry
from minference.ecs.entity import EntityRegistry, SqlEntityStorage, Entity, EntityBase
from minference.clients.utils import parse_json_string, msg_dict_to_oai, msg_dict_to_anthropic


logging.basicConfig(level=logging.INFO)

# 0) Remove the database file if it exists
db_file = "mydatabase.db"
if os.path.exists(db_file):
    os.remove(db_file)

# 1) Create your engine (here using SQLite as an example)
engine = create_engine(f"sqlite:///{db_file}", echo=False)

# 2) Drop all tables first to ensure clean slate
Base.metadata.drop_all(engine)

# Print table metadata before creating
for table in Base.metadata.sorted_tables:
    print(f"\nTable: {table.name}")
    for column in table.columns:
        print(f"  {column.name}: {column.type}")

# 3) Create all tables if needed
Base.metadata.create_all(engine)

# 4) Build a session factory
def session_factory():
    return sessionmaker(bind=engine)()

# 5) Map your domain classes to the corresponding ORM models
# Cast the mapping to the correct type to satisfy the type checker
entity_to_orm_map = cast(Dict[Type[Entity], Type[EntityBase]], ENTITY_MODEL_MAP)

# 6) Create the SQL storage object & tell the registry to use it
sql_storage = SqlEntityStorage(session_factory=session_factory, entity_to_orm_map=entity_to_orm_map)
EntityRegistry.use_storage(sql_storage)


async def main():
    load_dotenv()
    # Ensure both registries exist
    CallableRegistry()
    # We already switched EntityRegistry to the SQL storage above

    # Example request limits
    oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
    lite_llm_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
    anthropic_request_limits = RequestLimits(max_requests_per_minute=50, max_tokens_per_minute=20000)

    # Example orchestrator
    orchestrator = InferenceOrchestrator(
        oai_request_limits=oai_request_limits,
        litellm_request_limits=lite_llm_request_limits,
        anthropic_request_limits=anthropic_request_limits
    )

    # Example sets of chats for different providers
    lite_llm_model = "openai/NousResearch/Hermes-3-Llama-3.1-8B"
    anthropic_model = "claude-3-5-sonnet-latest"

    # Create a proper SystemPrompt instance
    system_string = SystemPrompt(
        name="joke_teller",
        content="You are a helpful assistant that tells programmer jokes."
    )

    # Example JSON schema for a structured tool
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

    # A small helper function to create chat threads
    def create_chats(client: LLMClient, model: str, response_formats: List[ResponseFormat], count=1) -> List[ChatThread]:
        chats: List[ChatThread] = []
        for response_format in response_formats:
            llm_config = LLMConfig(
                client=client, 
                model=model, 
                response_format=response_format,
                max_tokens=1000
            )
            for i in range(count):
                # Create system prompt first
                system_prompt = SystemPrompt(
                    name="joke_teller",
                    content="You are a helpful assistant that tells programmer jokes."
                )
                
                # Create chat thread
                chat = ChatThread(
                    system_prompt=system_prompt,
                    new_message=f"Tell me a programmer joke about the number {i}.",
                    llm_config=llm_config,
                    forced_output=structured_tool,
                    
                    sql_root=True  # Mark as root entity for SQL storage
                )
                chats.append(chat)
        return chats

    # Maybe you want just OpenAI tool-based requests
    openai_chats = create_chats(LLMClient.openai, "gpt-4o-mini", [ResponseFormat.tool], 1)

    # # Or more combos:
    # litellm_chats = create_chats(LLMClient.litellm, lite_llm_model, [ResponseFormat.tool], 2) + \
    #                 create_chats(LLMClient.litellm, lite_llm_model, [ResponseFormat.text], 2)
    # anthropic_chats = create_chats(LLMClient.anthropic, anthropic_model, [ResponseFormat.tool], 2) + \
    #                   create_chats(LLMClient.anthropic, anthropic_model, [ResponseFormat.text], 2)

    # all_chats = openai_chats + litellm_chats + anthropic_chats
    all_chats = openai_chats

    print("Running parallel completions...")
    start_time = time.time()
    completion_results = await orchestrator.run_parallel_ai_completion(all_chats)

    # We could do a second step or further steps
    for chat in all_chats:
        chat.new_message = "And why is it funny?"

    second_step_completion_results = await orchestrator.run_parallel_ai_completion(all_chats)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Completed in {total_time:.2f} seconds")

    # Print a mermaid graph of the first chat
    if all_chats:
        lineage_mermaid = EntityRegistry.get_lineage_mermaid(all_chats[0].lineage_id)
        print(lineage_mermaid)

    return all_chats


if __name__ == "__main__":
    # Run the async main
    all_chats = asyncio.run(main())
    # Optionally, do more with `all_chats`, or examine the DB contents via a Session
