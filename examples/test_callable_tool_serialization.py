"""
Test module for CallableTool serialization and deserialization.
Focuses on testing model_dump and model_validate functionality.
"""
from typing import List, Dict, Any
import json
import statistics
import asyncio
import logging
from uuid import UUID
from pydantic import BaseModel

from minference.lite.models import CallableTool
from minference.caregistry import CallableRegistry
from minference.entity import EntityRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize registries
EntityRegistry()
CallableRegistry()

# Test function with BaseModel input/output
STATS_ANALYZER = """
from typing import List
from pydantic import BaseModel
import statistics

class NumbersInput(BaseModel):
    numbers: List[float]
    round_to: int = 2

class Stats(BaseModel):
    mean: float
    std: float

def analyze_stats(input_data: NumbersInput) -> Stats:
    \"\"\"Calculate statistical measures using BaseModel input and output.\"\"\"
    return Stats(
        mean=round(statistics.mean(input_data.numbers), input_data.round_to),
        std=round(statistics.stdev(input_data.numbers), input_data.round_to)
    )
"""

# Test function with primitive types
MATH_FUNC = """
def multiply_numbers(x: float, y: float) -> float:
    \"\"\"Multiply two numbers together.\"\"\"
    return x * y
"""

async def test_tool_serialization():
    """Test serialization and deserialization of CallableTool."""
    print("\n=== Testing CallableTool Serialization ===")
    
    # Create initial tool
    tool = CallableTool.from_source(
        source=STATS_ANALYZER,
        name="analyze_stats",
        docstring="Statistical analysis tool"
    )
    
    # Test initial execution
    test_input = {"numbers": [1.0, 2.0, 3.0, 4.0, 5.0], "round_to": 3}
    result = await tool.aexecute(test_input)
    print(f"Initial tool execution result: {result}")
    
    # Instead of model_validate directly, use fork for new versions
    tool_dump = tool.model_dump()
    new_tool = tool.fork(**tool_dump)
    
    print(f"\nNew tool created with ID: {new_tool.id}")
    
    # Compare tool attributes
    print("\nComparing tool attributes:")
    print(f"- Same lineage: {tool.lineage_id == new_tool.lineage_id}")
    print(f"- Different IDs: {tool.id != new_tool.id}")
    print(f"- Parent is original: {new_tool.parent_id == tool.id}")
    print(f"- Same name: {tool.name == new_tool.name}")
    print(f"- Same input schema: {tool.input_schema == new_tool.input_schema}")
    print(f"- Same output schema: {tool.output_schema == new_tool.output_schema}")
    print(f"- Same callable text: {tool.callable_text == new_tool.callable_text}")
    
    # Test execution with new tool
    new_result = await new_tool.aexecute(test_input)
    print(f"\nNew tool execution result: {new_result}")
    
    # Verify results match
    assert str(result) == str(new_result), "Results don't match between original and deserialized tool"
    print("Results match between original and deserialized tool")
    
    # Now test creating a completely new tool instance with the same function
    print("\n=== Testing Creation of New Tool with Same Function ===")
    print("Creating new tool directly from model_dump of original tool...")
    
    # Get the tool dump and preserve schemas
    tool_dump = tool.model_dump()
    input_schema = tool_dump['input_schema']
    output_schema = tool_dump['output_schema']
    
    # Print schemas before direct creation
    print("\nSchema information before direct creation:")
    print("Input Schema:")
    print(json.dumps(input_schema, indent=2))
    print("\nOutput Schema:")
    print(json.dumps(output_schema, indent=2))
    
    # For direct creation test
    direct_tool = tool.fork(**tool_dump)  # Use fork instead of model_validate
    
    # Verify schemas match the preserved ones
    assert direct_tool.input_schema == input_schema, "Input schema changed during validation"
    assert direct_tool.output_schema == output_schema, "Output schema changed during validation"
    
    print(f"\nDirect tool created with ID: {direct_tool.id}")
    
    # Print schemas after direct creation
    print("\nSchema information after direct creation:")
    print("Input Schema:")
    print(json.dumps(direct_tool.input_schema, indent=2))
    print("\nOutput Schema:")
    print(json.dumps(direct_tool.output_schema, indent=2))
    
    # Compare all three tools
    print("\nComparing all three tools:")
    print("Original vs Model Validated:")
    print(f"- Share lineage: {tool.lineage_id == new_tool.lineage_id}")
    print(f"- Share function in registry: {CallableRegistry.get(tool.name) is CallableRegistry.get(new_tool.name)}")
    print(f"- Share input schema: {tool.input_schema == new_tool.input_schema}")
    print(f"- Share output schema: {tool.output_schema == new_tool.output_schema}")
    
    print("\nOriginal vs Direct Creation:")
    print(f"- Share lineage: {tool.lineage_id == direct_tool.lineage_id}")
    print(f"- Share function in registry: {CallableRegistry.get(tool.name) is CallableRegistry.get(direct_tool.name)}")
    print(f"- Share input schema: {tool.input_schema == direct_tool.input_schema}")
    print(f"- Share output schema: {tool.output_schema == direct_tool.output_schema}")
    
    # Test execution with direct tool
    direct_result = await direct_tool.aexecute(test_input)
    print(f"\nDirect tool execution result: {direct_result}")
    
    # Verify all results match
    assert str(result) == str(direct_result), "Results don't match between original and direct tool"
    print("All tools produce identical results")
    
    return tool, new_tool, direct_tool

async def test_tool_modification():
    """Test tool modification and re-serialization."""
    print("\n=== Testing Tool Modification ===")
    
    # Create initial tool
    tool = CallableTool.from_source(
        source=MATH_FUNC,
        name="multiply_numbers",
        docstring="Simple multiplication tool"
    )
    
    # Test initial execution
    test_input = {"x": 3.0, "y": 4.0}
    result = await tool.aexecute(test_input)
    print(f"Initial multiplication result: {result}")
    
    # Get model dump and examine
    tool_dump = tool.model_dump()
    print("\nInitial tool dump contents:")
    print(f"- Function name: {tool_dump['name']}")
    print(f"- Original callable text length: {len(tool_dump['callable_text'])}")
    print(f"- Input schema properties: {list(tool_dump['input_schema'].get('properties', {}).keys())}")
    
    # Modify the dump to change the function
    modified_source = """
def multiply_numbers(x: float, y: float) -> float:
    \"\"\"Modified multiplication with addition.\"\"\"
    return (x * y) + 1
"""
    
    # For tool modification test
    modified_tool = tool.fork(callable_text=modified_source)  # Use fork instead of model_validate
    
    # Compare the tools
    print("\nComparing original and modified tools:")
    print(f"- Same lineage: {tool.lineage_id == modified_tool.lineage_id}")
    print(f"- Different IDs: {tool.id != modified_tool.id}")
    print(f"- Same name: {tool.name == modified_tool.name}")
    print(f"- Same input schema: {tool.input_schema == modified_tool.input_schema}")
    print(f"- Same output schema: {tool.output_schema == modified_tool.output_schema}")
    print(f"- Different callable text: {tool.callable_text != modified_tool.callable_text}")
    
    # Test execution with modified tool
    modified_result = await modified_tool.aexecute(test_input)
    print(f"\nModified tool execution results:")
    print(f"- Original result: {result}")
    print(f"- Modified result: {modified_result}")
    print(f"- Difference: {float(modified_result['result']) - float(result['result'])}")  # Should be 1.0
    
    return tool, modified_tool

async def test_registry_interaction():
    """Test how serialization interacts with the registry."""
    print("\n=== Testing Registry Interaction ===")
    
    # Create and register initial tool
    tool = CallableTool.from_source(
        source=STATS_ANALYZER,
        name="analyze_stats_2",
        docstring="Another statistical analysis tool"
    )
    
    # Get model dump and examine registry state
    tool_dump = tool.model_dump()
    print("\nInitial tool state:")
    print(f"- Tool ID: {tool.id}")
    print(f"- Tool lineage ID: {tool.lineage_id}")
    
    print("\nInitial registry status:")
    registry_status = EntityRegistry.get_registry_status()
    print(f"- Registered functions: {len(registry_status['entities_by_type'])}")
    print(f"- Entity types: {registry_status['entities_by_type']}")
    print(f"- Version history:")
    print(f"  - First version: {registry_status['version_history']['first_version']}")
    print(f"  - Latest version: {registry_status['version_history']['latest_version']}")
    print(f"  - Version count: {registry_status['version_history']['version_count']}")
    print(f"- Total lineages: {registry_status['total_lineages']}")
    print(f"- Total versions: {registry_status['total_versions']}")
    
    # For registry interaction test
    new_tool = tool.fork(**tool_dump)  # Use fork instead of model_validate
    print(f"\nNew tool created with ID: {new_tool.id}")
    
    # Check registry status after new tool creation
    print("\nRegistry status after new tool creation:")
    new_status = EntityRegistry.get_registry_status()
    print(f"- Registered functions: {len(new_status['entities_by_type'])}")
    print(f"- Entity types: {new_status['entities_by_type']}")
    print(f"- Version history:")
    print(f"  - First version: {new_status['version_history']['first_version']}")
    print(f"  - Latest version: {new_status['version_history']['latest_version']}")
    print(f"  - Version count: {new_status['version_history']['version_count']}")
    print(f"- Total lineages: {new_status['total_lineages']}")
    print(f"- Total versions: {new_status['total_versions']}")
    
    # Verify both tools are in registry
    assert EntityRegistry.get(tool.id) is not None, "Original tool not in registry"
    assert EntityRegistry.get(new_tool.id) is not None, "New tool not in registry"
    print("\nTool registry verification:")
    print(f"- Original tool in registry: {EntityRegistry.get(tool.id) is not None}")
    print(f"- New tool in registry: {EntityRegistry.get(new_tool.id) is not None}")
    print(f"- Tools share lineage: {tool.lineage_id == new_tool.lineage_id}")
    print(f"- New tool parent is original: {new_tool.parent_id == tool.id}")
    
    return tool, new_tool

async def main():
    """Run all tests"""
    # Run serialization tests
    original_tool, deserialized_tool, direct_tool = await test_tool_serialization()
    
    # Run modification tests
    original_math, modified_math = await test_tool_modification()
    
    # Run registry tests
    registry_tool, new_registry_tool = await test_registry_interaction()
    
    print("\n=== All tests completed successfully ===")
    
    # Print final lineage trees
    print("\nOriginal tool lineage:")
    print(EntityRegistry.get_lineage_tree_sorted(original_tool.lineage_id))
    
    print("\nModified math tool lineage:")
    print(EntityRegistry.get_lineage_tree_sorted(modified_math.lineage_id))

if __name__ == "__main__":
    asyncio.run(main()) 