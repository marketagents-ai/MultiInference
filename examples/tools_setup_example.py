from typing import List, Dict, Tuple, Optional, Any
import json
import statistics
import asyncio
from uuid import UUID
from pydantic import BaseModel

from minference.lite.models import (
    CallableTool,
    MessageRole,
    ChatMessage,
    Entity,
    LLMConfig,
    LLMClient,
    ResponseFormat,
    SystemPrompt,
    ChatThread
)
from minference.lite.caregistry import CallableRegistry
from minference.lite.enregistry import EntityRegistry
from minference.lite.inference import InferenceOrchestrator

# Test Functions with different input/output patterns - each as a complete source string
BASEMODEL_ANALYZER = """
from typing import List
from pydantic import BaseModel
import statistics

class NumbersInput(BaseModel):
    numbers: List[float]
    round_to: int = 2

class Stats(BaseModel):
    mean: float
    std: float

def analyze_numbers_basemodel(input_data: NumbersInput) -> Stats:
    \"\"\"Calculate statistical measures using BaseModel input and output.\"\"\"
    return Stats(
        mean=round(statistics.mean(input_data.numbers), input_data.round_to),
        std=round(statistics.stdev(input_data.numbers), input_data.round_to)
    )
"""

REGULAR_ANALYZER = """
from typing import List
from pydantic import BaseModel
import statistics

class Stats(BaseModel):
    mean: float
    std: float

def analyze_numbers(numbers: List[float], round_to: int = 2) -> Stats:
    \"\"\"Calculate statistical measures with regular args returning BaseModel.\"\"\"
    return Stats(
        mean=round(statistics.mean(numbers), round_to),
        std=round(statistics.stdev(numbers), round_to)
    )
"""

DICT_ANALYZER = """
from typing import List, Dict
from pydantic import BaseModel
import statistics

class NumbersInput(BaseModel):
    numbers: List[float]
    round_to: int = 2

def analyze_numbers_dict(input_data: NumbersInput) -> Dict[str, float]:
    \"\"\"Calculate statistical measures using BaseModel input, returning dict.\"\"\"
    return {
        "mean": round(statistics.mean(input_data.numbers), input_data.round_to),
        "std": round(statistics.stdev(input_data.numbers), input_data.round_to)
    }
"""

TUPLE_ANALYZER = """
from typing import List, Tuple
import statistics

def analyze_numbers_tuple(numbers: List[float], round_to: int = 2) -> Tuple[float, float]:
    \"\"\"Calculate statistical measures returning tuple (mean, std).\"\"\"
    return (
        round(statistics.mean(numbers), round_to),
        round(statistics.stdev(numbers), round_to)
    )
"""

ASYNC_ANALYZER = """
import asyncio
import statistics
from typing import List, Dict

async def async_analyzer(numbers: List[float]) -> Dict[str, float]:
    \"\"\"Async statistical analysis.\"\"\"
    await asyncio.sleep(0.1)  # Simulate async work
    return {
        "mean": statistics.mean(numbers),
        "std": statistics.stdev(numbers)
    }
"""

ERROR_ANALYZER = """
from typing import List
from pydantic import BaseModel
import statistics

class Stats(BaseModel):
    mean: float
    std: float

def faulty_analyzer(numbers: List[float]) -> Stats:
    \"\"\"Analyzer that raises an error.\"\"\"
    if not numbers:
        raise ValueError("Empty input")
    return Stats(
        mean=statistics.mean(numbers),
        std=statistics.stdev(numbers)
    )
"""

class NumbersInput(BaseModel):
    numbers: List[float]
    round_to: int = 2

class Stats(BaseModel):
    mean: float
    std: float

async def test_type_patterns():
    """Test different input/output type patterns with the new CallableTool system"""
    print("\n=== Testing Type Patterns ===")
    test_numbers = [1.0, 2.0, 3.0, 4.0, 5.0]
    test_round_to = 3
    input_model = NumbersInput(numbers=test_numbers, round_to=test_round_to)
    
    # BaseModel -> BaseModel
    tool1 = CallableTool.from_source(
        source=BASEMODEL_ANALYZER,
        name="analyze_basemodel",
        docstring="Statistical analysis using BaseModel"
    )
    result1 = await tool1.aexecute(input_model.model_dump())
    print(f"BaseModel->BaseModel result: {result1}")
    verify_tool_integrity(tool1)
    
    # Regular args -> BaseModel
    tool2 = CallableTool.from_source(
        source=REGULAR_ANALYZER,
        name="analyze_regular",
        docstring="Statistical analysis with regular arguments"
    )
    result2 = await tool2.aexecute({
        "numbers": test_numbers,
        "round_to": test_round_to
    })
    print(f"Regular->BaseModel result: {result2}")
    verify_tool_integrity(tool2)
    
    # BaseModel -> Dict
    tool3 = CallableTool.from_source(
        source=DICT_ANALYZER,
        name="analyze_dict",
        docstring="Statistical analysis returning dictionary"
    )
    result3 = await tool3.aexecute(input_model.model_dump())
    print(f"BaseModel->Dict result: {result3}")
    verify_tool_integrity(tool3)
    
    # Regular args -> Tuple
    tool4 = CallableTool.from_source(
        source=TUPLE_ANALYZER,
        name="analyze_tuple",
        docstring="Statistical analysis returning tuple"
    )
    result4 = await tool4.aexecute({
        "numbers": test_numbers,
        "round_to": test_round_to
    })
    print(f"Regular->Tuple result: {result4}")
    verify_tool_integrity(tool4)
    
    return tool1, tool2, tool3, tool4

async def test_registry_operations():
    """Test registry functionality with the new system"""
    print("\n=== Testing Registry Operations ===")
    test_numbers = [1.0, 2.0, 3.0]
    input_model = NumbersInput(numbers=test_numbers)
    
    # Create initial tool
    tool = CallableTool.from_source(
        source=BASEMODEL_ANALYZER,
        name="custom_analyzer",
        docstring="Custom statistical analyzer"
    )
    
    result = await tool.aexecute(input_model.model_dump())
    print(f"Initial execution: {result}")
    
    # Test tool creation from source with modification
    UPDATED_ANALYZER = """
from typing import List
from pydantic import BaseModel
import statistics

class NumbersInput(BaseModel):
    numbers: List[float]
    round_to: int = 2

class Stats(BaseModel):
    mean: float
    std: float

def custom_analyzer(input_data: NumbersInput) -> Stats:
    \"\"\"Updated version with doubled mean.\"\"\"
    return Stats(
        mean=round(statistics.mean(input_data.numbers) * 2, input_data.round_to),
        std=round(statistics.stdev(input_data.numbers), input_data.round_to)
    )
"""
    
    tool_updated = CallableTool.from_source(
        source=UPDATED_ANALYZER,
        name="custom_analyzer",  # Same name to replace
        docstring="Updated statistical analyzer"
    )
    
    result_updated = await tool_updated.aexecute(input_model.model_dump())
    print(f"After update: {result_updated}")
    
    # Test deletion
    CallableRegistry.delete("custom_analyzer")
    try:
        await tool.aexecute(input_model.model_dump())
        raise AssertionError("Should have raised ValueError for deleted function")
    except ValueError as e:
        print(f"Caught deleted function execution: {e}")

async def test_tool_integration():
    """Test integration with ChatThread and LLM system"""
    print("\n=== Testing Tool Integration with ChatThread ===")
    
    # Create tool
    tool = CallableTool.from_source(
        source=BASEMODEL_ANALYZER,
        name="analyze_stats",
        docstring="Statistical analysis tool"
    )
    
    # Create system prompt
    system_prompt = SystemPrompt(
        name="stats_assistant",
        content="You are a statistical analysis assistant. Use the analyze_stats tool to help users understand their data."
    )
    
    # Create chat thread
    chat = ChatThread(
        system_prompt=system_prompt,
        new_message="Can you analyze these numbers: [1, 2, 3, 4, 5]?",
        llm_config=LLMConfig(
            client=LLMClient.openai,
            model="gpt-4",
            response_format=ResponseFormat.tool
        ),
        forced_output=tool
    )
    
    # Create orchestrator
    orchestrator = InferenceOrchestrator()
    
    # Run completion
    results = await orchestrator.run_parallel_ai_completion([chat])
    print(f"LLM Integration results: {results}")
    
    return chat, results

async def test_advanced_patterns():
    """Test advanced patterns including async operations"""
    print("\n=== Testing Advanced Patterns ===")
    
    # Test async analyzer
    tool = CallableTool.from_source(
        source=ASYNC_ANALYZER,
        name="async_analyzer",
        docstring="Async statistical analyzer"
    )
    
    result = await tool.aexecute({"numbers": [1, 2, 3, 4, 5]})
    print(f"Async tool result: {result}")
    
    # Test error analyzer
    error_tool = CallableTool.from_source(
        source=ERROR_ANALYZER,
        name="faulty_analyzer",
        docstring="Faulty analyzer for testing"
    )
    
    try:
        await error_tool.aexecute({"numbers": []})
    except ValueError as e:
        print(f"Successfully caught error: {e}")

def verify_tool_integrity(tool: CallableTool):
    """Verify tool properties and schema structure"""
    assert tool.name is not None, "Missing tool name"
    assert tool.input_schema is not None, "Missing input schema"
    assert tool.output_schema is not None, "Missing output schema"
    assert tool.callable_text is not None, "Missing callable text"
    
    # Verify schema structure
    assert "type" in tool.input_schema, "Input schema missing type"
    assert "properties" in tool.input_schema, "Input schema missing properties"
    assert "type" in tool.output_schema, "Output schema missing type"
    
    # Verify function is in registry
    assert CallableRegistry.get(tool.name) is not None, "Function not in registry"
    
    print(f"Tool {tool.name} passed integrity check")

async def main():
    """Run all tests"""
    # Initialize registries
    EntityRegistry()
    CallableRegistry()
    
    # Run tests
    await test_type_patterns()
    await test_registry_operations()
    await test_tool_integration()
    await test_advanced_patterns()
    
    print("\n=== All tests completed successfully ===")

if __name__ == "__main__":
    asyncio.run(main())