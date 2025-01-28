import pytest
from minference.lite.caregistry import CallableRegistry  # Update with your actual import path

# Singleton Pattern Tests
def test_singleton_behavior():
    """Test that CallableRegistry maintains singleton behavior."""
    # Create multiple instances
    registry1 = CallableRegistry()
    registry2 = CallableRegistry()
    registry3 = CallableRegistry()
    
    # Verify they're the same instance
    assert registry1 is registry2
    assert registry2 is registry3
    
    # Verify they share the same registry
    def test_func(x: int) -> int:
        return x
    
    CallableRegistry.register("test", test_func)
    assert "test" in registry1._registry
    assert "test" in registry2._registry
    assert "test" in registry3._registry

def test_shared_state():
    """Test that all instances share the same state."""
    # Clear any existing registrations
    registry1 = CallableRegistry()
    registry1._registry.clear()
    
    # Register through instance method
    def func1(x: int) -> int:
        return x
    
    registry1.register("func1", func1)
    
    # Register through class method
    def func2(x: int) -> int:
        return x * 2
    
    CallableRegistry.register("func2", func2)
    
    # Create new instance
    registry2 = CallableRegistry()
    
    # Verify all functions are available in all instances
    assert "func1" in registry1._registry
    assert "func2" in registry1._registry
    assert "func1" in registry2._registry
    assert "func2" in registry2._registry
    
    # Verify they're the same objects
    assert registry1._registry is registry2._registry

# Class Method vs Instance Method Tests
def test_class_vs_instance_methods():
    """Test that class methods and instance methods work identically."""
    def test_func(x: int) -> int:
        return x
    
    # Register using class method
    CallableRegistry.register("class_func", test_func)
    
    # Register using instance method
    registry = CallableRegistry()
    registry.register("instance_func", test_func)
    
    # Both should be accessible via either method
    assert CallableRegistry.get("class_func") is not None
    assert CallableRegistry.get("instance_func") is not None
    assert registry.get("class_func") is not None
    assert registry.get("instance_func") is not None

# Logging Persistence Tests
def test_logging_shared_state():
    """Test that logging state is shared across instances."""
    # Clear logs
    CallableRegistry.clear_logs()
    
    # Create instances
    registry1 = CallableRegistry()
    registry2 = CallableRegistry()
    
    def test_func(x: int) -> int:
        return x
    
    # Register using first instance
    registry1.register("test1", test_func)
    
    # Register using second instance
    registry2.register("test2", test_func)
    
    # Verify logs are shared
    logs = CallableRegistry.get_logs()
    assert "Successfully registered function: test1" in logs
    assert "Successfully registered function: test2" in logs
    
    # Verify logs are same for all instances
    assert registry1.get_logs() == registry2.get_logs()

# Basic Registration Tests
def test_basic_function_registration(registry, basic_functions):
    """Test registration of basic synchronous functions."""
    # Register using class method
    CallableRegistry.register("add", basic_functions["add"])
    
    # Verify registration both ways
    assert "add" in CallableRegistry._registry
    assert "add" in registry._registry
    assert CallableRegistry.get("add") == basic_functions["add"]
    assert registry.get("add") == basic_functions["add"]
    
    # Check function info using both methods
    info_class = CallableRegistry.get_info("add")
    info_instance = registry.get_info("add")
    
    assert info_class == info_instance
    assert info_class.name == "add"
    assert info_class.is_async is False
    assert "int" in info_class.input_schema["properties"]["x"]["type"]

def test_async_function_registration(registry, async_functions):
    """Test registration of async functions."""
    # Register using class method
    CallableRegistry.register("async_add", async_functions["async_add"])
    
    # Verify registration both ways
    assert "async_add" in CallableRegistry._registry
    assert "async_add" in registry._registry
    
    # Check function info
    info = CallableRegistry.get_info("async_add")
    assert info is not None
    assert info.name == "async_add"
    assert info.is_async is True

def test_model_function_registration(registry, model_functions):
    """Test registration of functions with Pydantic models."""
    # Register using class method
    CallableRegistry.register("process_user", model_functions["process_user"])
    
    # Verify registration
    info_class = CallableRegistry.get_info("process_user")
    info_instance = registry.get_info("process_user")
    
    assert info_class == info_instance
    assert "UserInput" in str(info_class.input_schema)
    assert "UserOutput" in str(info_class.output_schema)

@pytest.mark.import_heavy
def test_import_function_registration(registry, import_functions):
    """Test registration of functions with imports."""
    # Register using both class and instance methods
    CallableRegistry.register("numpy_func", import_functions["numpy_func"])
    registry.register("polars_func", import_functions["polars_func"])
    
    # Verify registrations in both instances
    assert "numpy_func" in CallableRegistry._registry
    assert "polars_func" in CallableRegistry._registry
    assert "numpy_func" in registry._registry
    assert "polars_func" in registry._registry

# Error Cases
def test_duplicate_registration(registry, basic_functions):
    """Test attempting to register the same function twice."""
    CallableRegistry.register("add", basic_functions["add"])
    
    # Try both class and instance method for duplicate
    with pytest.raises(ValueError) as exc_info:
        CallableRegistry.register("add", basic_functions["add"])
    assert "already registered" in str(exc_info.value)
    
    with pytest.raises(ValueError) as exc_info:
        registry.register("add", basic_functions["add"])
    assert "already registered" in str(exc_info.value)

def test_missing_type_hints():
    """Test registering a function without type hints."""
    def no_hints(x, y):  # Function without type hints
        return x + y
    
    with pytest.raises(ValueError) as exc_info:
        CallableRegistry.register("no_hints", no_hints)
    assert "must have type hints" in str(exc_info.value)

def test_invalid_return_hint():
    """Test registering a function with missing return hint."""
    def bad_return(x: int, y: int):  # Missing return hint
        return x + y
    
    with pytest.raises(ValueError) as exc_info:
        CallableRegistry.register("bad_return", bad_return)
    assert "return type hint" in str(exc_info.value)

# Update and Delete Tests
def test_function_update(registry, basic_functions):
    """Test updating a registered function."""
    # Initial registration with class method
    CallableRegistry.register("add", basic_functions["add"])
    
    def new_add(x: int, y: int) -> int:
        return x + y + 1  # Different implementation
    
    # Update with instance method
    registry.update("add", new_add)
    
    # Verify update in both class and instance
    assert CallableRegistry.get("add") == new_add
    assert registry.get("add") == new_add
    assert CallableRegistry.get("add")(1, 1) == 3
    assert registry.get("add")(1, 1) == 3

def test_function_deletion(registry, basic_functions):
    """Test deleting a registered function."""
    # Register with class method
    CallableRegistry.register("add", basic_functions["add"])
    assert "add" in CallableRegistry._registry
    
    # Delete with instance method
    registry.delete("add")
    assert "add" not in CallableRegistry._registry
    assert "add" not in registry._registry
    
    # Verify get returns None for both
    assert CallableRegistry.get("add") is None
    assert registry.get("add") is None

def test_delete_nonexistent(registry):
    """Test attempting to delete a non-registered function."""
    with pytest.raises(ValueError) as exc_info:
        CallableRegistry.delete("nonexistent")
    assert "not found" in str(exc_info.value)
    
    with pytest.raises(ValueError) as exc_info:
        registry.delete("nonexistent")
    assert "not found" in str(exc_info.value)

# Registry Status Tests
def test_registry_status(registry, basic_functions):
    """Test getting registry status information."""
    # Register using both methods
    CallableRegistry.register("add", basic_functions["add"])
    registry.register("multiply", basic_functions["multiply"])
    
    # Get status from both
    status_class = CallableRegistry.get_registry_status()
    status_instance = registry.get_registry_status()
    
    # Verify they're identical
    assert status_class == status_instance
    assert status_class["total_functions"] == 2
    assert "add" in status_class["registered_functions"]
    assert "multiply" in status_class["registered_functions"]
    assert len(status_class["function_signatures"]) == 2

# Logging Tests
def test_registration_logging(registry, basic_functions, assert_logs):
    """Test that registration operations are properly logged."""
    # Clear logs first
    CallableRegistry.clear_logs()
    
    # Test successful registration logging
    CallableRegistry.register("add", basic_functions["add"])
    assert_logs(registry, "Successfully registered function: add")
    
    # Test failed registration logging
    try:
        CallableRegistry.register("add", basic_functions["add"])
    except ValueError:
        pass
    assert_logs(registry, "Registration failed: Function 'add' already registered")

# Text Registration Tests
def test_register_from_text(registry):
    """Test registering a function from text representation."""
    func_text = """
def text_func(x: int, y: int) -> int:
    return x + y
"""
    # Register using class method
    CallableRegistry.register_from_text("text_func", func_text)
    assert "text_func" in CallableRegistry._registry
    assert "text_func" in registry._registry
    
    # Verify function works in both contexts
    func_class = CallableRegistry.get("text_func")
    func_instance = registry.get("text_func")
    assert func_class(1, 2) == 3
    assert func_instance(1, 2) == 3

def test_register_lambda_from_text(registry):
    """Test registering a lambda function from text."""
    lambda_text = "lambda x: x * 2"
    
    # Register using instance method
    registry.register_from_text("lambda_func", lambda_text)
    
    # Verify function works in both contexts
    class_func = CallableRegistry.get("lambda_func")
    instance_func = registry.get("lambda_func")
    assert class_func(2) == 4
    assert instance_func(2) == 4

# Thread Safety Tests
@pytest.mark.asyncio
async def test_concurrent_registration():
    """Test concurrent registration operations."""
    import asyncio
    
    async def register_func(name: str, value: int):
        def func(x: int) -> int:
            return x + value
        CallableRegistry.register(name, func)
        await asyncio.sleep(0.1)  # Simulate work
    
    # Perform concurrent registrations
    await asyncio.gather(
        register_func("func1", 1),
        register_func("func2", 2),
        register_func("func3", 3)
    )
    
    # Verify all registrations succeeded in both contexts
    registry = CallableRegistry()
    assert all(f"func{i}" in CallableRegistry._registry for i in range(1, 4))
    assert all(f"func{i}" in registry._registry for i in range(1, 4))

# Error State Consistency
def test_error_state_consistency():
    """Test that error states don't corrupt the shared registry."""
    registry1 = CallableRegistry()
    registry2 = CallableRegistry()
    
    def valid_func(x: int) -> int:
        return x
    
    # Register valid function with class method
    CallableRegistry.register("valid", valid_func)
    
    # Try to register invalid function with instance
    def invalid_func(x):  # Missing return hint
        return x
    
    with pytest.raises(ValueError):
        registry2.register("invalid", invalid_func)
    
    # Verify registry state is consistent across all contexts
    assert "valid" in CallableRegistry._registry
    assert "valid" in registry1._registry
    assert "valid" in registry2._registry
    assert "invalid" not in CallableRegistry._registry
    assert "invalid" not in registry1._registry
    assert "invalid" not in registry2._registry