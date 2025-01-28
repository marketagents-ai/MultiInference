# conftest.py
import pytest
import asyncio
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import logging
from pydantic import BaseModel, EmailStr

from minference.lite.caregistry import CallableRegistry  # Update with your actual import path

# Custom test markers
def pytest_configure(config):
    """Configure custom markers."""
    markers = [
        "slow: marks tests as slow",
        "import_heavy: marks tests that involve heavy imports",
        "async_test: marks async tests",
        "integration: marks integration tests",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)

# Basic fixtures
@pytest.fixture
def registry():
    """Provide a fresh registry instance."""
    reg = CallableRegistry()
    reg.clear_logs()  # Start with clean logs
    yield reg
    # Cleanup
    for func_name in list(reg._registry.keys()):
        reg.delete(func_name)
    reg.clear_logs()

@pytest.fixture
def basic_functions():
    """Provide a set of basic test functions."""
    def add(x: int, y: int) -> int:
        """Simple addition function"""
        return x + y

    def multiply(x: float, y: float) -> float:
        """Simple multiplication function"""
        return x * y

    return {
        "add": add,
        "multiply": multiply
    }

@pytest.fixture
def async_functions():
    """Provide a set of async test functions."""
    async def async_add(x: int, y: int) -> int:
        """Async addition function"""
        await asyncio.sleep(0.1)  # Simulate async work
        return x + y

    async def async_multiply(x: float, y: float) -> float:
        """Async multiplication function"""
        await asyncio.sleep(0.1)  # Simulate async work
        return x * y

    return {
        "async_add": async_add,
        "async_multiply": async_multiply
    }

@pytest.fixture
def model_functions():
    """Provide functions that work with Pydantic models."""
    class UserInput(BaseModel):
        name: str
        age: int
        email: str

    class UserOutput(BaseModel):
        id: int
        name: str
        age: int
        email: str
        is_adult: bool

    def process_user(user: UserInput) -> UserOutput:
        """Process user data with models"""
        return UserOutput(
            id=hash(user.email),
            name=user.name,
            age=user.age,
            email=user.email,
            is_adult=user.age >= 18
        )

    return {
        "process_user": process_user,
        "UserInput": UserInput,
        "UserOutput": UserOutput
    }

# In conftest.py
@pytest.fixture
def import_functions():
    """Provide functions with various import patterns."""
    def numpy_func(x: float, y: float) -> float:
        """Function with numpy import"""
        import numpy as np
        return float(np.multiply(x, y))

    def polars_func(data: Dict[str, list]) -> Dict[str, float]:
        """Function with polars import"""
        import polars as pl
        
        # Input validation
        if not data or not all(isinstance(v, list) for v in data.values()):
            raise ValueError("Input must be a dict of lists")
            
        df = pl.DataFrame(data)
        
        # Ensure we have numeric data to aggregate
        if df.shape[0] == 0 or df.shape[1] == 0:
            return {"mean": 0.0, "std": 0.0}
            
        # Calculate aggregations safely
        means = df.select(pl.all().mean()).row(0)
        stds = df.select(pl.all().std()).row(0)
        
        # Calculate overall statistics
        overall_mean = sum(means) / len(means) if means else 0.0
        overall_std = sum(stds) / len(stds) if stds else 0.0
        
        return {
            "mean": float(overall_mean),
            "std": float(overall_std)
        }

    return {
        "numpy_func": numpy_func,
        "polars_func": polars_func
    }

@pytest.fixture
def error_functions():
    """Provide functions that generate various errors."""
    def import_error(x: int) -> int:
        """Function with import error"""
        import non_existent_package #type: ignore
        return x

    def runtime_error(x: int) -> int:
        """Function with runtime error"""
        raise RuntimeError("Simulated error")

    def type_error(x: int) -> int:
        """Function with type error"""
        return "not an integer"  # Type error

    return {
        "import_error": import_error,
        "runtime_error": runtime_error,
        "type_error": type_error
    }

@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return {
        "simple_data": {
            "x": 10,
            "y": 20
        },
        "user_data": {
            "name": "Test User",
            "age": 25,
            "email": "test@example.com"
        },
        "dataframe_data": {
            "A": [1, 2, 3, 4, 5],
            "B": [2, 4, 6, 8, 10]
        }
    }

# Utility fixtures
@pytest.fixture
def assert_logs():
    """Fixture to assert log content."""
    def _assert_logs(registry: CallableRegistry, expected_content: str, level: str = "INFO"):
        logs = registry.get_logs()
        assert expected_content in logs, f"Expected '{expected_content}' in logs"
        assert f"- {level} -" in logs, f"Expected {level} level log entry"
    return _assert_logs

@pytest.fixture
def benchmark_executor():
    """Fixture for benchmarking function execution."""
    async def _benchmark(func: Callable, input_data: Dict[str, Any], iterations: int = 1000):
        start = asyncio.get_event_loop().time()
        for _ in range(iterations):
            await func(**input_data)
        end = asyncio.get_event_loop().time()
        return (end - start) / iterations
    return _benchmark