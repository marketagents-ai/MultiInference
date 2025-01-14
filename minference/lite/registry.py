"""
Registry module for managing callable tools and their schemas.

This module provides a global registry for managing callable functions along with 
their input/output schemas. It includes functionality for registering functions,
deriving schemas from type hints, and executing registered functions safely both
synchronously and asynchronously.

Main components:
- CallableRegistry: Singleton registry for managing functions
- Schema helpers: Functions for deriving and validating JSON schemas
- Registration helpers: Functions for safely registering callables
- Execution helpers: Functions for safely executing registered callables
"""

from typing import Dict, Any, List, Optional, Literal, Union, Tuple, Callable, TypeAlias, Protocol, TypeVar, Awaitable, get_type_hints
from pydantic import ValidationError, create_model, BaseModel
import json
from ast import literal_eval
import sys
import libcst as cst
from inspect import signature, iscoroutinefunction
from dataclasses import dataclass
from datetime import datetime
import logging
from io import StringIO
import asyncio

# Type aliases and protocols
JsonDict: TypeAlias = Dict[str, Any]
SchemaType: TypeAlias = Dict[str, Any]
RegistryType: TypeAlias = Dict[str, Callable]

T = TypeVar('T')

class AsyncCallable(Protocol):
    """Protocol for async callable objects"""
    async def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class SyncCallable(Protocol):
    """Protocol for sync callable objects"""
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

AnyCallable = Union[AsyncCallable, SyncCallable]

@dataclass
class RegistryInfo:
    """Information about a registered function"""
    name: str
    signature: str
    doc: Optional[str]
    input_schema: SchemaType
    output_schema: SchemaType
    is_async: bool

def is_async_callable(func: Callable) -> bool:
    """Check if a callable is async."""
    return iscoroutinefunction(func) or hasattr(func, '__await__')

async def ensure_async(func: AnyCallable, *args: Any, **kwargs: Any) -> Any:
    """
    Ensure a function is executed asynchronously.
    Wraps sync functions in asyncio.to_thread.
    """
    if is_async_callable(func):
        return await func(*args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)

class CallableRegistry:
    """Global registry for tool callables"""
    _instance = None
    _registry: RegistryType = {}
    
    # Setup logging
    _log_stream = StringIO()
    _logger = logging.getLogger('CallableRegistry')
    _handler = logging.StreamHandler(_log_stream)
    _handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)

    def __new__(cls) -> 'CallableRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._logger.info("Initializing new CallableRegistry instance")
        return cls._instance

    @classmethod
    def get_logs(cls) -> str:
        """Get all logs as text."""
        return cls._log_stream.getvalue()
    
    @classmethod
    def clear_logs(cls) -> None:
        """Clear all logs."""
        cls._log_stream.truncate(0)
        cls._log_stream.seek(0)
        cls._logger.info("Logs cleared")

    @classmethod
    def set_log_level(cls, level: Union[int, str]) -> None:
        """Set the logging level."""
        cls._logger.setLevel(level)
        cls._logger.info(f"Log level set to {level}")

    @classmethod
    def register(cls, name: str, func: Callable) -> None:
        """Register a new callable with validation."""
        cls._logger.info(f"Attempting to register function: {name}")
        
        if name in cls._registry:
            cls._logger.error(f"Registration failed: Function '{name}' already registered")
            raise ValueError(f"Function '{name}' already registered. Use update() to replace.")
        
        try:
            cls._validate_type_hints(name, func)
            cls._registry[name] = func
            cls._logger.info(f"Successfully registered function: {name}")
        except Exception as e:
            cls._logger.error(f"Registration failed for {name}: {str(e)}")
            raise
    
    @classmethod
    def _validate_type_hints(cls, name: str, func: Callable) -> None:
        """Validate that a function has proper type hints."""
        cls._logger.debug(f"Validating type hints for function: {name}")
        
        type_hints = get_type_hints(func)
        if not type_hints:
            cls._logger.error(f"Validation failed: Function '{name}' has no type hints")
            raise ValueError(f"Function '{name}' must have type hints")
        if 'return' not in type_hints:
            cls._logger.error(f"Validation failed: Function '{name}' has no return type hint")
            raise ValueError(f"Function '{name}' must have a return type hint")
        
        cls._logger.debug(f"Type hints valid for function: {name}")

    @classmethod
    def register_from_text(cls, name: str, func_text: str) -> None:
        """Register a function from its text representation with safety checks."""
        cls._logger.info(f"Attempting to register function from text: {name}")
        
        if name in cls._registry:
            cls._logger.error(f"Registration failed: Function '{name}' already registered")
            raise ValueError(f"Function '{name}' already registered. Use update().")
        
        try:
            func = cls._parse_function_text(name, func_text)
            cls._validate_type_hints(name, func)
            cls._registry[name] = func
            cls._logger.info(f"Successfully registered function from text: {name}")
        except Exception as e:
            cls._logger.error(f"Failed to register function from text: {str(e)}")
            raise ValueError(f"Failed to parse function: {str(e)}")

    @staticmethod
    def _parse_function_text(name: str, func_text: str) -> Callable:
        """Parse function text into a callable with safety checks."""
        # Handle lambdas
        if func_text.strip().startswith('lambda'):
            wrapper_text = f"""
def {name}(x: float) -> float:
    \"\"\"Wrapped lambda function\"\"\"
    func = {func_text}
    return func(x)
"""
            func_text = wrapper_text

        try:
            module = cst.parse_module(func_text)
            namespace = {
                'float': float, 'int': int, 'str': str, 'bool': bool,
                'list': list, 'dict': dict, 'tuple': tuple,
                'List': List, 'Dict': Dict, 'Tuple': Tuple,
                'Optional': Optional, 'Union': Union, 'Any': Any,
                'BaseModel': BaseModel
            }
            
            exec(module.code, namespace)
            
            if func_text.strip().startswith('lambda'):
                return namespace[name]
            return namespace[func_text.split('def ')[1].split('(')[0].strip()]
            
        except Exception as e:
            raise ValueError(f"Failed to parse function: {str(e)}")

    @classmethod
    def update(cls, name: str, func: Callable) -> None:
        """Update an existing callable with validation."""
        cls._logger.info(f"Attempting to update function: {name}")
        try:
            cls._validate_type_hints(name, func)
            cls._registry[name] = func
            cls._logger.info(f"Successfully updated function: {name}")
        except Exception as e:
            cls._logger.error(f"Update failed for {name}: {str(e)}")
            raise
    
    @classmethod
    def delete(cls, name: str) -> None:
        """Delete a callable from registry."""
        cls._logger.info(f"Attempting to delete function: {name}")
        if name not in cls._registry:
            cls._logger.error(f"Deletion failed: Function '{name}' not found")
            raise ValueError(f"Function '{name}' not found in registry.")
        del cls._registry[name]
        cls._logger.info(f"Successfully deleted function: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        """Get a registered callable by name."""
        cls._logger.debug(f"Retrieving function: {name}")
        return cls._registry.get(name)
    
    @classmethod
    def get_info(cls, name: str) -> Optional[RegistryInfo]:
        """Get detailed information about a registered function."""
        cls._logger.debug(f"Retrieving info for function: {name}")
        
        func = cls.get(name)
        if not func:
            cls._logger.debug(f"No info available: Function '{name}' not found")
            return None
            
        return RegistryInfo(
            name=name,
            signature=str(signature(func)),
            doc=func.__doc__,
            input_schema=derive_input_schema(func),
            output_schema=derive_output_schema(func),
            is_async=is_async_callable(func)
        )

    @classmethod
    def get_registry_status(cls) -> JsonDict:
        """Get current status of the registry."""
        cls._logger.debug("Retrieving registry status")
        return {
            "total_functions": len(cls._registry),
            "registered_functions": list(cls._registry.keys()),
            "function_signatures": {
                name: str(signature(func))
                for name, func in cls._registry.items()
            }
        }
    
    @classmethod
    def execute(cls, name: str, input_data: JsonDict) -> JsonDict:
        """
        Execute a registered callable synchronously.
        
        Args:
            name: Name of function to execute
            input_data: Input data for function
            
        Returns:
            JsonDict containing the execution result
            
        Raises:
            ValueError: If function not found or execution fails
        """
        cls._logger.info(f"Attempting sync execution of function: {name}")
        try:
            result = execute_callable(name, input_data, registry=cls())
            cls._logger.info(f"Successfully executed {name} synchronously")
            return result
        except Exception as e:
            cls._logger.error(f"Sync execution failed for {name}: {str(e)}")
            raise
        
    @classmethod
    async def aexecute(cls, name: str, input_data: JsonDict) -> JsonDict:
        """Execute a registered callable asynchronously."""
        cls._logger.info(f"Attempting async execution of function: {name}")
        try:
            result = await aexecute_callable(name, input_data, registry=cls())
            cls._logger.info(f"Successfully executed {name} asynchronously")
            return result
        except Exception as e:
            cls._logger.error(f"Async execution failed for {name}: {str(e)}")
            raise
    

# Registration helper functions
def register_default_callable(name: str, func: Callable, registry: Optional[CallableRegistry] = None) -> None:
    """Register a callable from default tools collection."""
    if registry is None:
        registry = CallableRegistry()
    try:
        registry.register(name, func)
    except ValueError:
        pass  # Already registered

def register_from_text_with_validation(
    name: str, 
    func_text: str,
    allow_literal_eval: bool = False,
    registry: Optional[CallableRegistry] = None
) -> None:
    """Register a function from text with safety validation."""
    if registry is None:
        registry = CallableRegistry()
        
    if not allow_literal_eval:
        raise ValueError("allow_literal_eval must be True to register from text")
        
    try:
        registry.register_from_text(name=name, func_text=func_text)
    except Exception as e:
        raise ValueError(f"Could not register function from text: {str(e)}")

# Schema helper functions
def derive_input_schema(func: Callable) -> SchemaType:
    """Derive JSON schema from function type hints."""
    type_hints = get_type_hints(func)
    sig = signature(func)
    
    if 'return' not in type_hints:
        raise ValueError(f"Function {func.__name__} must have a return type hint")
    
    first_param = next(iter(sig.parameters.values()))
    first_param_type = type_hints.get(first_param.name)
    
    if (isinstance(first_param_type, type) and 
        issubclass(first_param_type, BaseModel)):
        return first_param_type.model_json_schema()
    
    input_fields = {}
    for param_name, param in sig.parameters.items():
        if param_name not in type_hints:
            raise ValueError(f"Parameter {param_name} must have a type hint")
        
        if param.default is param.empty:
            input_fields[param_name] = (type_hints[param_name], ...)
        else:
            input_fields[param_name] = (type_hints[param_name], param.default)

    InputModel = create_model(f"{func.__name__}Input", **input_fields)
    return InputModel.model_json_schema()

def derive_output_schema(func: Callable) -> SchemaType:
    """Derive output JSON schema from function return type."""
    type_hints = get_type_hints(func)
    
    if 'return' not in type_hints:
        raise ValueError(f"Function {func.__name__} must have a return type hint")
    
    output_type = type_hints['return']
    if isinstance(output_type, type) and issubclass(output_type, BaseModel):
        OutputModel = output_type
    else:
        OutputModel = create_model(f"{func.__name__}Output", result=(output_type, ...))
    
    return OutputModel.model_json_schema()

def validate_schema_compatibility(
    derived_schema: SchemaType,
    provided_schema: SchemaType
) -> None:
    """Validate that provided schema matches derived schema."""
    derived_required = set(derived_schema.get("required", []))
    provided_required = set(provided_schema.get("required", []))
    if derived_required != provided_required:
        raise ValueError(
            f"Schema mismatch: Required properties don't match.\n"
            f"Derived: {derived_required}\n"
            f"Provided: {provided_required}"
        )

    derived_props = derived_schema.get("properties", {})
    provided_props = provided_schema.get("properties", {})
    
    for prop_name, prop_schema in derived_props.items():
        if prop_name not in provided_props:
            raise ValueError(f"Missing property '{prop_name}' in provided schema")
        provided_type = provided_props[prop_name].get("type")
        derived_type = prop_schema.get("type")
        if provided_type != derived_type:
            raise ValueError(
                f"Property '{prop_name}' type mismatch.\n"
                f"Derived: {derived_type}\n"
                f"Provided: {provided_type}"
            )

    extra_props = set(provided_props.keys()) - set(derived_props.keys())
    if extra_props:
        raise ValueError(f"Extra properties in provided schema: {extra_props}")

def execute_callable(
    name: str,
    input_data: JsonDict,
    registry: Optional[CallableRegistry] = None
) -> JsonDict:
    """Execute a registered callable with input data."""
    if registry is None:
        registry = CallableRegistry()
        
    callable_func = registry.get(name)
    if not callable_func:
        raise ValueError(
            f"Function '{name}' not found in registry. "
            f"Available: {list(registry._registry.keys())}"
        )
    
    try:
        sig = signature(callable_func)
        type_hints = get_type_hints(callable_func)
        first_param = next(iter(sig.parameters.values()))
        param_type = type_hints.get(first_param.name)
        
        # Handle input based on parameter type
        if (isinstance(param_type, type) and 
            issubclass(param_type, BaseModel)):
            model_input = param_type.model_validate(input_data)
            response = callable_func(model_input)
        else:
            response = callable_func(**input_data)
        
        # Handle response serialization
        if isinstance(response, BaseModel):
            return json.loads(response.model_dump_json())
        return {"result": response}
        
    except Exception as e:
        raise ValueError(f"Error executing {name}: {str(e)}") from e

async def aexecute_callable(
    name: str,
    input_data: JsonDict,
    registry: Optional[CallableRegistry] = None
) -> JsonDict:
    """
    Execute a registered callable asynchronously with input data.
    Handles both async and sync functions.
    
    Args:
        name: Name of function to execute
        input_data: Input data for function
        registry: Optional registry instance (uses global if None)
        
    Returns:
        JsonDict containing the execution result
        
    Raises:
        ValueError: If function not found or execution fails
    """
    if registry is None:
        registry = CallableRegistry()
        
    callable_func = registry.get(name)
    if not callable_func:
        raise ValueError(
            f"Function '{name}' not found in registry. "
            f"Available: {list(registry._registry.keys())}"
        )
    
    try:
        sig = signature(callable_func)
        type_hints = get_type_hints(callable_func)
        first_param = next(iter(sig.parameters.values()))
        param_type = type_hints.get(first_param.name)
        
        # Prepare input
        if (isinstance(param_type, type) and 
            issubclass(param_type, BaseModel)):
            model_input = param_type.model_validate(input_data)
            input_arg = model_input
        else:
            input_arg = input_data
            
        # Execute function based on its type
        if iscoroutinefunction(callable_func):
            if isinstance(input_arg, BaseModel):
                response = await callable_func(input_arg)
            else:
                response = await callable_func(**input_arg)
        else:
            # Run sync function in executor to avoid blocking
            if isinstance(input_arg, BaseModel):
                response = await asyncio.to_thread(callable_func, input_arg)
            else:
                response = await asyncio.to_thread(callable_func, **input_arg)
        
        # Handle response serialization
        if isinstance(response, BaseModel):
            return json.loads(response.model_dump_json())
        return {"result": response}
        
    except Exception as e:
        raise ValueError(f"Error executing {name}: {str(e)}") from e

__all__ = [
    'CallableRegistry',
    'RegistryInfo',
    'JsonDict',
    'SchemaType',
    'RegistryType',
    'AsyncCallable',
    'SyncCallable',
    'AnyCallable',
    'register_default_callable',
    'register_from_text_with_validation',
    'derive_input_schema',
    'derive_output_schema',
    'validate_schema_compatibility',
    'execute_callable',
    'aexecute_callable',
    'is_async_callable',
    'ensure_async'
]