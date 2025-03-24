import asyncio
from dotenv import load_dotenv
from minference.threads.inference import InferenceOrchestrator, RequestLimits
from minference.threads.models import ChatMessage, ChatThread, LLMConfig, CallableTool, LLMClient,ResponseFormat, SystemPrompt, StructuredTool, Usage,GeneratedJsonObject
from typing import Literal, List
from minference.ecs.caregistry import CallableRegistry
import time
from minference.clients.utils import msg_dict_to_oai, msg_dict_to_anthropic, parse_json_string
from minference.ecs.entity import EntityRegistry
import os
import logging
import json
load_dotenv()
EntityRegistry()
CallableRegistry()
oai_request_limits = RequestLimits(max_requests_per_minute=10000, max_tokens_per_minute=200000000)
lite_llm_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
anthropic_request_limits = RequestLimits(max_requests_per_minute=1500, max_tokens_per_minute=2000000)
lite_llm_model = "deephermes-3-llama-3-8b-preview-mlx"
lite_llm_model = "qwen2.5-7b-instruct-mlx"
vllm_model = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
vllm_model = "Qwen/Qwen2.5-7B-Instruct"
vllm_request_limits = RequestLimits(max_requests_per_minute=500000, max_tokens_per_minute=200000000)
orchestrator = InferenceOrchestrator(oai_request_limits=oai_request_limits, litellm_request_limits=lite_llm_request_limits, anthropic_request_limits=anthropic_request_limits, vllm_request_limits=vllm_request_limits)
EntityRegistry.set_inference_orchestrator(orchestrator)
EntityRegistry.set_tracing_enabled(False)

#load json from all_monsters.json
all_monsters = json.load(open(r"/Users/tommasofurlanello/Documents/Dev/MarketInference/examples/all_monsters.json"))

monster_manual = {}
for monster in all_monsters['monster']:
    #monster is a dict we want ot dump it to json string
    monster_manual[monster["name"]] = json.dumps(monster)




simpler_monster_schema={
  "type": "object",
  "title": "Monster",
  "description": "Simplified D&D 5e monster stats with focus on numerical values",
  "properties": {
    "name": {
      "type": "string",
      "description": "The monster's name"
    },
    "source": {
      "type": ["string", "null"],
      "description": "Source book"
    },
    "size": {
      "type": ["string", "null"],
      "enum": ["T", "S", "M", "L", "H", "G"],
      "description": "Size code"
    },
    "type": {
      "type": ["string", "null"],
      "description": "Creature type"
    },
    "ac": {
      "type": ["integer", "null"],
      "description": "Armor class"
    },
    "hp": {
      "type": ["integer", "null"],
      "description": "Hit points"
    },
    "hd_count": {
      "type": ["integer", "null"],
      "description": "Hit dice count"
    },
    "hd_size": {
      "type": ["integer", "null"],
      "description": "Hit dice size"
    },
    "hd_bonus": {
      "type": ["integer", "null"],
      "description": "Hit dice modifier"
    },
    "str": {
      "type": ["integer", "null"],
      "description": "Strength"
    },
    "dex": {
      "type": ["integer", "null"],
      "description": "Dexterity"
    },
    "con": {
      "type": ["integer", "null"],
      "description": "Constitution"
    },
    "int": {
      "type": ["integer", "null"],
      "description": "Intelligence"
    },
    "wis": {
      "type": ["integer", "null"],
      "description": "Wisdom"
    },
    "cha": {
      "type": ["integer", "null"],
      "description": "Charisma"
    },
    "speed_walk": {
      "type": ["integer", "null"],
      "description": "Walking speed"
    },
    "speed_fly": {
      "type": ["integer", "null"],
      "description": "Flying speed"
    },
    "speed_swim": {
      "type": ["integer", "null"],
      "description": "Swimming speed"
    },
    "speed_climb": {
      "type": ["integer", "null"],
      "description": "Climbing speed"
    },
    "speed_burrow": {
      "type": ["integer", "null"],
      "description": "Burrowing speed"
    },
    "darkvision": {
      "type": ["integer", "null"],
      "description": "Darkvision range"
    },
    "blindsight": {
      "type": ["integer", "null"],
      "description": "Blindsight range"
    },
    "tremorsense": {
      "type": ["integer", "null"],
      "description": "Tremorsense range"
    },
    "truesight": {
      "type": ["integer", "null"],
      "description": "Truesight range"
    },
    "passive_perception": {
      "type": ["integer", "null"],
      "description": "Passive perception"
    },
    "cr": {
      "type": ["number", "string", "null"],
      "description": "Challenge rating"
    },
    "xp": {
      "type": ["integer", "null"],
      "description": "XP value"
    },
    "prof_bonus": {
      "type": ["integer", "null"],
      "description": "Proficiency bonus"
    },
    "multiattack_count": {
      "type": ["integer", "null"],
      "description": "Number of attacks in multiattack"
    },
    "attack1_bonus": {
      "type": ["integer", "null"],
      "description": "Attack 1 to-hit bonus"
    },
    "attack1_reach": {
      "type": ["integer", "null"],
      "description": "Attack 1 reach"
    },
    "attack1_range": {
      "type": ["integer", "null"],
      "description": "Attack 1 range"
    },
    "attack1_dice_count": {
      "type": ["integer", "null"],
      "description": "Attack 1 damage dice count"
    },
    "attack1_dice_size": {
      "type": ["integer", "null"],
      "description": "Attack 1 damage dice size"
    },
    "attack1_damage_bonus": {
      "type": ["integer", "null"],
      "description": "Attack 1 damage bonus"
    },
    "attack2_bonus": {
      "type": ["integer", "null"],
      "description": "Attack 2 to-hit bonus"
    },
    "attack2_reach": {
      "type": ["integer", "null"],
      "description": "Attack 2 reach"
    },
    "attack2_range": {
      "type": ["integer", "null"],
      "description": "Attack 2 range"
    },
    "attack2_dice_count": {
      "type": ["integer", "null"],
      "description": "Attack 2 damage dice count"
    },
    "attack2_dice_size": {
      "type": ["integer", "null"],
      "description": "Attack 2 damage dice size"
    },
    "attack2_damage_bonus": {
      "type": ["integer", "null"],
      "description": "Attack 2 damage bonus"
    },
    "save_dc1": {
      "type": ["integer", "null"],
      "description": "Save DC 1"
    },
    "save_dc2": {
      "type": ["integer", "null"],
      "description": "Save DC 2"
    },
    "legendary_actions_count": {
      "type": ["integer", "null"],
      "description": "Number of legendary actions per round"
    },
    "spellcasting_level": {
      "type": ["integer", "null"],
      "description": "Spellcasting level"
    }
  },
  "required": [
    "name", "source", "size", "type", "ac", "hp", "hd_count", "hd_size", "hd_bonus",
    "str", "dex", "con", "int", "wis", "cha", 
    "speed_walk", "speed_fly", "speed_swim", "speed_climb", "speed_burrow",
    "darkvision", "blindsight", "tremorsense", "truesight", "passive_perception",
    "cr", "xp", "prof_bonus", "multiattack_count",
    "attack1_bonus", "attack1_reach", "attack1_range", "attack1_dice_count", 
    "attack1_dice_size", "attack1_damage_bonus",
    "attack2_bonus", "attack2_reach", "attack2_range", "attack2_dice_count", 
    "attack2_dice_size", "attack2_damage_bonus",
    "save_dc1", "save_dc2", "legendary_actions_count", "spellcasting_level"
  ],
  "additionalProperties": False
}

monster_extractor = StructuredTool(name="monster_extractor",json_schema=simpler_monster_schema, post_validate_schema=False)

system_prompt = SystemPrompt(content="You are a epidemiologist expert that is studying monsters in an alternative universe that can extract information from a text and convert it into a structured output. You will be given a text and you will need to extract the information and convert it into a structured output. You will be given a text and you will need to extract the information and convert it into a structured output.", name= "monster_extractor")

llm_config_local = LLMConfig(client=LLMClient.vllm, model=vllm_model, response_format=ResponseFormat.tool,max_tokens=4000)
llm_config=LLMConfig(client=LLMClient.openai, model="gpt-4o-mini", response_format=ResponseFormat.tool,max_tokens=4000)

thread = ChatThread(
    system_prompt=system_prompt,
    new_message="",
    llm_config=llm_config,
    forced_output=monster_extractor,
    use_schema_instruction=True
)
thread_id = thread.id

local_thread = ChatThread(
    system_prompt=system_prompt,
    new_message="",
    llm_config=llm_config_local,
    forced_output=monster_extractor,
    use_schema_instruction=True
)
local_thread_id = local_thread.id
            
thread_list = []
thread_dict = {}
i = 0
for monster_name, monster_data in monster_manual.items():
    if i >2000:
        break
    base_thread  = EntityRegistry.get(thread_id)
    assert base_thread is not None
    base_thread.fork(force=True,**{"new_message":f"Please study in detail the monster {monster_data}"})
    thread_list.append(base_thread.id)
    thread_dict[monster_name] = base_thread.id
    i += 1


local_thread_list = []
local_thread_dict = {}
j = 0
for monster_name, monster_data in monster_manual.items():
    if j>i:
      base_thread  = EntityRegistry.get(local_thread_id)
      assert base_thread is not None
      base_thread.fork(force=True,**{"new_message":f"Please study in detail the monster {monster_data}"})
      local_thread_list.append(base_thread.id)
      local_thread_dict[monster_name] = base_thread.id
    j += 1


threads = EntityRegistry.get_many(thread_list)
local_threads = EntityRegistry.get_many(local_thread_list)
print(threads[0].new_message)

test_threads = threads[:100]
test_threads = local_threads
import time

start_time = time.time()

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Get the number of CPU cores
num_cores = multiprocessing.cpu_count()

async def run_threads(threads):
    # Create a ThreadPoolExecutor with max_workers set to number of CPU cores
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=num_cores) as pool:
        outputs = await orchestrator.run_parallel_ai_completion(threads)
    return outputs

processed_outputs = asyncio.run(run_threads(test_threads))
end_time = time.time()

print(processed_outputs[-1].json_object)
print(f"Time taken: {end_time - start_time} seconds for a total of {len(processed_outputs)} threads with an average of {(end_time - start_time)/len(processed_outputs)}  seconds per thread with a total of cpu cores {num_cores}")
