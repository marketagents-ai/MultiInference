import asyncio
import logging
import os
import time

from dotenv import load_dotenv
from typing import Literal, List, Dict, Type, cast

# SQLModel / database imports
from sqlalchemy import create_engine, inspect
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
from minference.ecs.entity import Entity
from minference.ecs.storage import SqlEntityStorage, EntityBase, BaseEntitySQL
from minference.ecs.enregistry import EntityRegistry
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

# CRITICAL: Directly create the base_entity table since it's not being registered properly
# Use direct SQL approach to create the table if it doesn't exist
from sqlalchemy import text

# First, check if the base_entity table already exists
inspector = inspect(engine)
if not 'base_entity' in inspector.get_table_names():
    # Create the base_entity table directly with SQL
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE base_entity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ecs_id CHAR(32) NOT NULL,
            lineage_id CHAR(32) NOT NULL,
            parent_id CHAR(32),
            created_at DATETIME NOT NULL,
            old_ids TEXT NOT NULL,
            class_name VARCHAR(255) NOT NULL,
            data TEXT,
            entity_type VARCHAR(50) NOT NULL DEFAULT 'base_entity'
        )
        """))
        conn.commit()
    print("Created base_entity table directly with SQL")

# Import RequestLimitsSQL to ensure it's included in metadata too
from minference.threads.sql_models import RequestLimitsSQL

# Print table metadata before creating
for table in Base.metadata.sorted_tables:
    print(f"\nTable: {table.name}")
    for column in table.columns:
        print(f"  {column.name}: {column.type}")

# Create the tables
Base.metadata.create_all(engine)

# Verify the required tables exist
with engine.connect() as conn:
    # Check base_entity table
    result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='base_entity'"))
    base_entity_exists = result.fetchone() is not None
    print(f"base_entity table exists: {base_entity_exists}")
    
    # Check request_limits table
    result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='request_limits'"))
    request_limits_exists = result.fetchone() is not None
    print(f"request_limits table exists: {request_limits_exists}")
    
    if not base_entity_exists:
        print("WARNING: base_entity table still not created - creating it directly")
        # Create the base_entity table directly with SQL as a last resort
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS base_entity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ecs_id CHAR(32) NOT NULL,
            lineage_id CHAR(32) NOT NULL,
            parent_id CHAR(32),
            created_at DATETIME NOT NULL,
            old_ids TEXT NOT NULL,
            class_name VARCHAR(255) NOT NULL,
            data TEXT,
            entity_type VARCHAR(50) NOT NULL DEFAULT 'base_entity'
        )
        """))
        conn.commit()
        
    if not request_limits_exists:
        print("WARNING: request_limits table not created! Make sure RequestLimits entity is registered.")
        
    # Verify again after our fix attempt
    all_tables = [row[0] for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()]
    print(f"All tables in database: {all_tables}")

# 4) Build a session factory
def session_factory():
    return sessionmaker(bind=engine)()

# 5) Map your domain classes to the corresponding ORM models
# Cast the mapping to the correct type to satisfy the type checker
entity_to_orm_map = cast(Dict[Type[Entity], Type[EntityBase]], ENTITY_MODEL_MAP)

# CRITICAL: Make sure Entity base class is mapped to BaseEntitySQL for fallback
# Without this, entities without specific SQL models won't be able to be stored
entity_to_orm_map[Entity] = cast(Type[EntityBase], BaseEntitySQL)

# CONFIRM RequestLimits is in the mapping (CRITICAL for our fix)
from minference.threads.models import RequestLimits
if RequestLimits not in entity_to_orm_map:
    print("WARNING: RequestLimits not found in ENTITY_MODEL_MAP! Adding it now.")
    entity_to_orm_map[RequestLimits] = RequestLimitsSQL
else:
    print(f"✅ Good: RequestLimits found in ENTITY_MODEL_MAP (mapped to {entity_to_orm_map[RequestLimits].__name__})")

# Print the whole mapping for debugging
print("Entity to ORM mapping:")
for entity_class, orm_class in entity_to_orm_map.items():
    print(f" - {entity_class.__name__} -> {orm_class.__name__}")

# 7) Create the SQL storage object & tell the registry to use it
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
