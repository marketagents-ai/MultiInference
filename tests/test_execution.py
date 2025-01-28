# test_execution.py
import pytest
import asyncio
from typing import Any, Dict

# Synchronous Execution Tests
def test_basic_sync_execution(registry, basic_functions):
    """Test basic synchronous function execution."""
    # Register function
    registry.register("add", basic_functions["add"])
    
    # Execute
    result = registry.execute("add", {"x": 1, "y": 2})
    assert result["result"] == 3

def test_sync_execution_with_defaults(registry):
    """Test execution with default parameters."""
    def func_with_defaults(x: int, y: int = 10) -> int:
        return x + y
        
    registry.register("with_defaults", func_with_defaults)
    
    # Test with and without default parameter
    result1 = registry.execute("with_defaults", {"x": 1, "y": 2})
    result2 = registry.execute("with_defaults", {"x": 1})
    
    assert result1["result"] == 3
    assert result2["result"] == 11

def test_sync_execution_with_models(registry, model_functions):
    """Test execution with Pydantic models."""
    registry.register("process_user", model_functions["process_user"])
    
    test_data = {
        "name": "Test User",
        "age": 25,
        "email": "test@example.com"
    }
    
    result = registry.execute("process_user", test_data)
    assert result["name"] == "Test User"
    assert result["age"] == 25
    assert result["is_adult"] is True

# In test_execution.py, modify the test:

@pytest.mark.import_heavy
def test_sync_execution_with_imports(registry, import_functions):
    """Test execution of functions with imports."""
    registry.register("numpy_func", import_functions["numpy_func"])
    registry.register("polars_func", import_functions["polars_func"])
    
    # Test numpy function
    np_result = registry.execute("numpy_func", {"x": 2.0, "y": 3.0})
    assert np_result["result"] == 6.0
    
    # Test polars function
    pl_data = {
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    }
    pl_result = registry.execute("polars_func", {"data": pl_data})
    assert "mean" in pl_result["result"]
    assert "std" in pl_result["result"]
    # We can also test the actual values
    assert pl_result["result"]["mean"] == 3.5
    assert pl_result["result"]["std"] == 1.0

# Asynchronous Execution Tests
@pytest.mark.asyncio
async def test_basic_async_execution(registry, async_functions):
    """Test basic asynchronous function execution."""
    registry.register("async_add", async_functions["async_add"])
    
    result = await registry.aexecute("async_add", {"x": 1, "y": 2})
    assert result["result"] == 3

@pytest.mark.asyncio
async def test_sync_function_async_execution(registry, basic_functions):
    """Test executing synchronous function through async interface."""
    registry.register("add", basic_functions["add"])
    
    result = await registry.aexecute("add", {"x": 1, "y": 2})
    assert result["result"] == 3

@pytest.mark.asyncio
async def test_concurrent_execution(registry, async_functions):
    """Test concurrent execution of multiple functions."""
    registry.register("async_add", async_functions["async_add"])
    registry.register("async_multiply", async_functions["async_multiply"])
    
    # Execute multiple functions concurrently
    results = await asyncio.gather(
        registry.aexecute("async_add", {"x": 1, "y": 2}),
        registry.aexecute("async_multiply", {"x": 3, "y": 4}),
        registry.aexecute("async_add", {"x": 5, "y": 6})
    )
    
    assert results[0]["result"] == 3  # 1 + 2
    assert results[1]["result"] == 12  # 3 * 4
    assert results[2]["result"] == 11  # 5 + 6

# Error Handling Tests
def test_execution_not_found(registry):
    """Test executing non-existent function."""
    with pytest.raises(ValueError) as exc:
        registry.execute("nonexistent", {"x": 1})
    assert "not found" in str(exc.value)

def test_execution_invalid_params(registry, basic_functions):
    """Test executing with invalid parameters."""
    registry.register("add", basic_functions["add"])
    
    with pytest.raises(ValueError) as exc:
        registry.execute("add", {"x": 1})  # Missing y parameter
    assert "missing" in str(exc.value).lower()

@pytest.mark.asyncio
async def test_async_execution_error(registry):
    """Test error handling in async execution."""
    async def error_func(x: int) -> int:
        raise ValueError("Simulated error")
        
    registry.register("error_func", error_func)
    
    with pytest.raises(ValueError) as exc:
        await registry.aexecute("error_func", {"x": 1})
    assert "Simulated error" in str(exc.value)

# Resource Management Tests
@pytest.mark.asyncio
async def test_execution_cleanup(registry):
    """Test resource cleanup after execution."""
    async def resource_func(x: int) -> int:
        try:
            return x
        finally:
            # Simulate resource cleanup
            pass
            
    registry.register("resource_func", resource_func)
    result = await registry.aexecute("resource_func", {"x": 1})
    assert result["result"] == 1

@pytest.mark.slow
def test_large_payload_execution(registry):
    """Test execution with large data payload."""
    def large_data_func(data: Dict[str, Any]) -> int:
        return len(str(data))
        
    registry.register("large_data", large_data_func)
    
    # Create large payload
    large_payload = {"data": [i for i in range(10000)]}
    
    result = registry.execute("large_data", large_payload)
    assert isinstance(result["result"], int)
    assert result["result"] > 0

# Performance Tests
@pytest.mark.benchmark
def test_execution_performance(registry, benchmark):
    """Test execution performance."""
    def perf_func(x: int) -> int:
        return x * 2
        
    registry.register("perf_func", perf_func)
    
    def run_execution():
        return registry.execute("perf_func", {"x": 42})
        
    result = benchmark(run_execution)
    assert result["result"] == 84

# Schema Validation During Execution
def test_execution_schema_validation(registry):
    """Test schema validation during execution."""
    from pydantic import BaseModel
    
    class InputModel(BaseModel):
        x: int
        y: float
        
    class OutputModel(BaseModel):
        result: float
        
    def schema_func(data: InputModel) -> OutputModel:
        return OutputModel(result=data.x + data.y)
        
    registry.register("schema_func", schema_func)
    
    # Valid input
    result = registry.execute("schema_func", {"x": 1, "y": 2.5})
    assert result["result"] == 3.5
    
    # Invalid input
    with pytest.raises(ValueError) as exc:
        registry.execute("schema_func", {"x": "invalid", "y": 2.5})
    assert "validation error" in str(exc.value).lower()