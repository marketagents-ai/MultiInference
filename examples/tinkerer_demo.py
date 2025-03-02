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
anthropic_request_limits = RequestLimits(max_requests_per_minute=50, max_tokens_per_minute=20000)

orchestrator = InferenceOrchestrator(oai_request_limits=oai_request_limits, litellm_request_limits=lite_llm_request_limits, anthropic_request_limits=anthropic_request_limits)
EntityRegistry.set_inference_orchestrator(orchestrator)
EntityRegistry.set_tracing_enabled(False)

#load json from all_monsters.json
all_monsters = json.load(open(r"/Users/tommasofurlanello/Documents/Dev/MarketInference/examples/all_monsters.json"))

monster_manual = {}
for monster in all_monsters['monster']:
    monster_manual[monster["name"]] = monster



monster_schema={
  "type": "object",
  "title": "Monster",
  "description": "Essential D&D 5e monster stats with numerical combat values",
  "properties": {
    "name": {
      "type": "string",
      "description": "The monster's name"
    },
    "source": {
      "type": ["string", "null"],
      "description": "Source book or material"
    },
    "size": {
      "type": ["string", "null"],
      "enum": ["T", "S", "M", "L", "H", "G"],
      "description": "Size category (Tiny to Gargantuan)"
    },
    "type": {
      "type": ["string", "null"],
      "enum": [
        "aberration", "beast", "celestial", "construct", "dragon", 
        "elemental", "fey", "fiend", "giant", "humanoid",
        "monstrosity", "ooze", "plant", "undead"
      ],
      "description": "Creature type"
    },
    "armor_class": {
      "type": ["integer", "null"],
      "description": "Base armor class"
    },
    "hit_points": {
      "type": ["integer", "null"],
      "description": "Average hit points"
    },
    "hit_dice": {
      "type": "object",
      "description": "Hit dice details",
      "properties": {
        "count": {
          "type": ["integer", "null"],
          "description": "Number of dice (e.g., 3 in 3d8+6)"
        },
        "size": {
          "type": ["integer", "null"],
          "description": "Size of dice (e.g., 8 in 3d8+6)"
        },
        "modifier": {
          "type": ["integer", "null"],
          "description": "Modifier (e.g., 6 in 3d8+6)"
        }
      },
      "required": ["count", "size", "modifier"],
      "additionalProperties": False
    },
    "ability_scores": {
      "type": "object",
      "description": "The six ability scores",
      "properties": {
        "str": {
          "type": ["integer", "null"],
          "description": "Strength score"
        },
        "dex": {
          "type": ["integer", "null"],
          "description": "Dexterity score"
        },
        "con": {
          "type": ["integer", "null"],
          "description": "Constitution score"
        },
        "int": {
          "type": ["integer", "null"],
          "description": "Intelligence score"
        },
        "wis": {
          "type": ["integer", "null"],
          "description": "Wisdom score"
        },
        "cha": {
          "type": ["integer", "null"],
          "description": "Charisma score"
        }
      },
      "required": ["str", "dex", "con", "int", "wis", "cha"],
      "additionalProperties": False
    },
    "speeds": {
      "type": "object",
      "description": "Movement speeds in feet",
      "properties": {
        "walk": {
          "type": "integer",
          "description": "Walking speed"
        },
        "fly": {
          "type": ["integer", "null"],
          "description": "Flying speed"
        },
        "swim": {
          "type": ["integer", "null"],
          "description": "Swimming speed"
        },
        "climb": {
          "type": ["integer", "null"],
          "description": "Climbing speed"
        },
        "burrow": {
          "type": ["integer", "null"],
          "description": "Burrowing speed"
        }
      },
      "required": ["walk", "fly", "swim", "climb", "burrow"],
      "additionalProperties": False
    },
    "senses": {
      "type": "object",
      "description": "Special senses in feet",
      "properties": {
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
          "description": "Passive Perception score"
        }
      },
      "required": ["darkvision", "blindsight", "tremorsense", "truesight", "passive_perception"],
      "additionalProperties": False
    },
    "resistances": {
      "type": "array",
      "description": "Damage resistances",
      "items": {
        "type": "string",
        "enum": [
          "acid", "bludgeoning", "cold", "fire", "force", "lightning", 
          "necrotic", "piercing", "poison", "psychic", "radiant", 
          "slashing", "thunder"
        ]
      }
    },
    "immunities": {
      "type": "array",
      "description": "Damage immunities",
      "items": {
        "type": "string",
        "enum": [
          "acid", "bludgeoning", "cold", "fire", "force", "lightning", 
          "necrotic", "piercing", "poison", "psychic", "radiant", 
          "slashing", "thunder"
        ]
      }
    },
    "vulnerabilities": {
      "type": "array",
      "description": "Damage vulnerabilities",
      "items": {
        "type": "string",
        "enum": [
          "acid", "bludgeoning", "cold", "fire", "force", "lightning", 
          "necrotic", "piercing", "poison", "psychic", "radiant", 
          "slashing", "thunder"
        ]
      }
    },
    "condition_immunities": {
      "type": "array",
      "description": "Condition immunities",
      "items": {
        "type": "string",
        "enum": [
          "blinded", "charmed", "deafened", "exhaustion", "frightened",
          "grappled", "incapacitated", "invisible", "paralyzed", 
          "petrified", "poisoned", "prone", "restrained", "stunned", 
          "unconscious"
        ]
      }
    },
    "special_traits": {
      "type": "array",
      "description": "Special abilities and traits",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "Name of the trait"
          },
          "description": {
            "type": "string",
            "description": "Description of the trait"
          }
        },
        "required": ["name", "description"],
        "additionalProperties": False
      }
    },
    "actions": {
      "type": "array",
      "description": "Actions the monster can take",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "Name of the action"
          },
          "type": {
            "type": ["string", "null"],
            "enum": ["melee", "ranged", "both", "other"],
            "description": "Type of attack"
          },
          "attack_bonus": {
            "type": ["integer", "null"],
            "description": "Attack bonus to hit"
          },
          "reach": {
            "type": ["integer", "null"],
            "description": "Reach in feet for melee attacks"
          },
          "range": {
            "type": ["integer", "null"],
            "description": "Normal range for ranged attacks"
          },
          "damage_dice_count": {
            "type": ["integer", "null"],
            "description": "Number of dice (e.g., 2 in 2d6)"
          },
          "damage_dice_size": {
            "type": ["integer", "null"],
            "description": "Size of dice (e.g., 6 in 2d6)"
          },
          "damage_bonus": {
            "type": ["integer", "null"],
            "description": "Static damage bonus (e.g., 3 in 2d6+3)"
          },
          "damage_type": {
            "type": ["string", "null"],
            "enum": [
              "acid", "bludgeoning", "cold", "fire", "force", "lightning", 
              "necrotic", "piercing", "poison", "psychic", "radiant", 
              "slashing", "thunder"
            ],
            "description": "Type of damage dealt"
          },
          "save_dc": {
            "type": ["integer", "null"],
            "description": "Save DC if applicable"
          },
          "save_ability": {
            "type": ["string", "null"],
            "enum": ["str", "dex", "con", "int", "wis", "cha"],
            "description": "Ability for saving throw"
          }
        },
        "required": ["name", "type", "attack_bonus", "reach", "range", 
                    "damage_dice_count", "damage_dice_size", "damage_bonus", 
                    "damage_type", "save_dc", "save_ability"],
        "additionalProperties": False
      }
    },
    "legendary_actions": {
      "type": "array",
      "description": "Legendary actions",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "Name of the legendary action"
          },
          "description": {
            "type": "string",
            "description": "Description of the legendary action"
          },
          "cost": {
            "type": "integer",
            "description": "Action cost (usually 1-3)"
          }
        },
        "required": ["name", "description", "cost"],
        "additionalProperties": False
      }
    },
    "challenge_rating": {
      "type": ["number", "string", "null"],
      "description": "Challenge rating"
    },
    "xp": {
      "type": ["integer", "null"],
      "description": "XP value"
    },
    "languages": {
      "type": "array",
      "description": "Languages known",
      "items": {
        "type": "string"
      }
    }
  },
  "required": [
    "name", "source", "size", "type", "armor_class", "hit_points", 
    "hit_dice", "ability_scores", "speeds", "senses", "resistances", 
    "immunities", "vulnerabilities", "condition_immunities", "special_traits", 
    "actions", "legendary_actions", "challenge_rating", "xp", "languages"
  ],
  "additionalProperties": False
}

monster_extractor = StructuredTool(name="monster_extractor",json_schema=monster_schema, post_validate_schema=False)

system_prompt = SystemPrompt(content="You are a epidemiologist expert that is studying monsters in an alternative universe that can extract information from a text and convert it into a structured output. You will be given a text and you will need to extract the information and convert it into a structured output. You will be given a text and you will need to extract the information and convert it into a structured output.", name= "monster_extractor")
llm_config=LLMConfig(client=LLMClient.openai, model="gpt-4o-mini", response_format=ResponseFormat.structured_output,max_tokens=4000)

llm_config=LLMConfig(client=LLMClient.openai, model="gpt-4o-mini", response_format=ResponseFormat.structured_output,max_tokens=4000)

thread = ChatThread(
    system_prompt=system_prompt,
    new_message="",
    llm_config=llm_config,
    forced_output=monster_extractor,
    use_schema_instruction=True
)
thread_id = thread.id
            
thread_list = []
thread_dict = {}
for monster_name, monster_data in monster_manual.items():
    base_thread  = EntityRegistry.get(thread_id)
    assert base_thread is not None
    base_thread.fork(force=True,**{"new_message":f"Please study in detail the monster {monster_data}"})
    thread_list.append(base_thread.id)
    thread_dict[monster_name] = base_thread.id

threads = EntityRegistry.get_many(thread_list)

print(threads[0].new_message)

test_threads = threads[:100]
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
