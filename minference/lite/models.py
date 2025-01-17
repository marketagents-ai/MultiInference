"""
Entity models with registry integration and serialization support.

This module provides:
1. Base Entity class for registry-integrated, serializable objects
2. CallableTool implementation for managing executable functions
3. Serialization and persistence capabilities
4. Registry integration for both entities and callables
"""
from typing import Dict, Any, Optional, ClassVar, Type, TypeVar, List, Generic, Callable
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, model_validator
from pathlib import Path
import json
from datetime import datetime
import inspect

from minference.lite.caregistry import (
    CallableRegistry,
    derive_input_schema,
    derive_output_schema,
    validate_schema_compatibility,
)
from minference.lite.enregistry import EntityRegistry

T = TypeVar('T', bound='Entity')

class Entity(BaseModel):
    """
    Base class for registry-integrated, serializable entities.
    
    This class provides:
    1. Integration with EntityRegistry for persistence and retrieval
    2. Basic serialization interface for saving/loading
    3. Common entity attributes and operations
    
    Subclasses are responsible for:
    1. Implementing custom serialization if needed (_custom_serialize/_custom_deserialize)
    2. Handling any nested entities or complex relationships
    3. Managing entity-specific validation and business logic
    
    All entities are immutable - modifications require creating new instances.
    """
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this entity instance"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when this entity was created"
    )
    
    model_config = {
        "frozen": True,  # Ensures immutability
        "arbitrary_types_allowed": True
    }
    
    @model_validator(mode='after')
    def register_entity(self) -> 'Entity':
        """Register this entity instance in the registry."""
        registry = EntityRegistry()
        registry._logger.debug(f"{self.__class__.__name__}({self.id}): Registering entity")
        
        try:
            registry.register(self)
            registry._logger.debug(f"{self.__class__.__name__}({self.id}): Successfully registered")
        except Exception as e:
            registry._logger.error(f"{self.__class__.__name__}({self.id}): Registration failed - {str(e)}")
            raise ValueError(f"Entity registration failed: {str(e)}") from e
            
        return self
    
    def _custom_serialize(self) -> Dict[str, Any]:
        """
        Custom serialization hook for subclasses.
        
        Override this method to add custom serialization logic.
        The result will be included in the serialized output under 'custom_data'.
        
        Returns:
            Dict containing any custom serialized data
        """
        return {}
        
    @classmethod
    def _custom_deserialize(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Custom deserialization hook for subclasses.
        
        Override this method to handle custom deserialization logic.
        The input comes from the 'custom_data' field in serialized data.
        
        Args:
            data: Custom data from serialized entity
            
        Returns:
            Dict of deserialized fields to include in entity initialization
        """
        return {}
    
    def save(self, path: Path) -> None:
        """
        Save this entity instance to a file.
        
        Args:
            path: Path where to save the entity
            
        Raises:
            IOError: If saving fails
        """
        registry = EntityRegistry()
        registry._logger.debug(f"{self.__class__.__name__}({self.id}): Saving to {path}")
        
        try:
            # Get base serialization
            data = self.model_dump()
            
            # Add metadata
            metadata = {
                "entity_type": self.__class__.__name__,
                "schema_version": "1.0",
                "saved_at": datetime.utcnow().isoformat()
            }
            
            # Get custom data
            custom_data = self._custom_serialize()
            
            # Combine all
            serialized = {
                "metadata": metadata,
                "data": data,
                "custom_data": custom_data
            }
            
            # Save
            with open(path, 'w') as f:
                json.dump(serialized, f, indent=2)
                
            registry._logger.debug(f"{self.__class__.__name__}({self.id}): Successfully saved")
            
        except Exception as e:
            registry._logger.error(f"{self.__class__.__name__}({self.id}): Save failed - {str(e)}")
            raise IOError(f"Failed to save entity: {str(e)}") from e
    
    @classmethod
    def load(cls: Type[T], path: Path) -> T:
        """
        Load an entity instance from a file.
        
        Args:
            path: Path to the saved entity
            
        Returns:
            Loaded entity instance
            
        Raises:
            IOError: If loading fails
            ValueError: If data validation fails
        """
        registry = EntityRegistry()
        registry._logger.debug(f"{cls.__name__}: Loading from {path}")
        
        try:
            # Load file
            with open(path) as f:
                serialized = json.load(f)
                
            # Verify entity type
            metadata = serialized["metadata"]
            if metadata["entity_type"] != cls.__name__:
                raise ValueError(
                    f"Entity type mismatch. File contains {metadata['entity_type']}, "
                    f"expected {cls.__name__}"
                )
                
            # Get base and custom data
            base_data = serialized["data"]
            custom_data = cls._custom_deserialize(serialized.get("custom_data", {}))
            
            # Create instance
            instance = cls(**{**base_data, **custom_data})
            registry._logger.debug(f"{cls.__name__}({instance.id}): Successfully loaded")
            return instance
            
        except Exception as e:
            registry._logger.error(f"{cls.__name__}: Load failed - {str(e)}")
            raise IOError(f"Failed to load entity: {str(e)}") from e
            
    @classmethod
    def get(cls: Type[T], entity_id: UUID) -> Optional[T]:
        """Get an entity instance from the registry."""
        return EntityRegistry().get(entity_id, expected_type=cls)
        
    @classmethod
    def list_all(cls: Type[T]) -> List[T]:
        """List all entities of this type."""
        return EntityRegistry().list_by_type(cls)
        
    @classmethod
    def get_many(cls: Type[T], entity_ids: List[UUID]) -> List[T]:
        """Get multiple entities by their IDs."""
        return EntityRegistry().get_many(entity_ids, expected_type=cls)


class CallableTool(Entity):
    """
    An immutable callable tool that can be registered and executed with schema validation.
    
    Inherits from Entity for registry integration and serialization support.
    The tool is registered in:
    - CallableRegistry: For the actual function registration and execution
    - EntityRegistry: For the tool metadata and versioning (handled by parent)
    
    Any modifications require creating new instances with new UUIDs.
    """    
    name: str = Field(
        description="Registry name for the callable function",
        min_length=1
    )
    
    docstring: Optional[str] = Field(
        default=None,
        description="Documentation describing the tool's purpose and usage"
    )
    
    input_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema defining valid input parameters"
    )
    
    output_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema defining the expected output format"
    )
    
    strict_schema: bool = Field(
        default=True,
        description="Whether to enforce strict schema validation"
    )
    
    callable_text: Optional[str] = Field(
        default=None,
        description="Source code of the callable function"
    )

    model_config = {
        "frozen": True,  # Ensures immutability
        "arbitrary_types_allowed": True,
        "json_schema_extra": {
            "examples": [
                {
                    "name": "multiply",
                    "docstring": "Multiplies two numbers together",
                    "callable_text": "def multiply(x: float, y: float) -> float:\n    return x * y"
                }
            ]
        }
    }
    
    @model_validator(mode='after')
    def validate_schemas_and_callable(self) -> 'CallableTool':
        """
        Validates the tool's schemas and ensures its callable is registered.
        
        This validator:
        1. Checks/registers the callable in CallableRegistry
        2. Extracts callable text if not provided
        3. Derives and validates input/output schemas
        """
        ca_registry = CallableRegistry()
        ca_registry._logger.debug(f"CallableTool({self.id}): Validating function '{self.name}'")
        
        func = ca_registry.get(self.name)
        if not func:
            if not self.callable_text:
                ca_registry._logger.error(f"CallableTool({self.id}): No callable or text provided for '{self.name}'")
                raise ValueError(f"No callable found in registry for '{self.name}' and no callable_text provided")
                
            try:
                ca_registry.register_from_text(self.name, self.callable_text)
                func = ca_registry.get(self.name)
            except Exception as e:
                ca_registry._logger.error(f"CallableTool({self.id}): Function registration failed for '{self.name}'")
                raise ValueError(f"Failed to register callable: {str(e)}")
                
        # Store text representation if not provided
        if not self.callable_text:
            try:
                self.callable_text = inspect.getsource(func)
            except (TypeError, OSError):
                self.callable_text = str(func)
        
        # Derive and validate schemas
        derived_input = derive_input_schema(func)
        derived_output = derive_output_schema(func)
        
        # Validate against provided schemas if they exist
        if self.input_schema:
            try:
                validate_schema_compatibility(derived_input, self.input_schema)
            except ValueError as e:
                ca_registry._logger.error(f"CallableTool({self.id}): Input schema validation failed for '{self.name}'")
                raise ValueError(f"Input schema validation failed: {str(e)}")
        else:
            self.input_schema = derived_input
            
        if self.output_schema:
            try:
                validate_schema_compatibility(derived_output, self.output_schema)
            except ValueError as e:
                ca_registry._logger.error(f"CallableTool({self.id}): Output schema validation failed for '{self.name}'")
                raise ValueError(f"Output schema validation failed: {str(e)}")
        else:
            self.output_schema = derived_output
            
        ca_registry._logger.debug(f"CallableTool({self.id}): Successfully validated function '{self.name}'")
        return self
    
    def _custom_serialize(self) -> Dict[str, Any]:
        """Serialize the callable-specific data."""
        return {
            "schemas": {
                "input": self.input_schema,
                "output": self.output_schema
            }
        }
    
    @classmethod
    def _custom_deserialize(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize the callable-specific data."""
        schemas = data.get("schemas", {})
        return {
            "input_schema": schemas.get("input", {}),
            "output_schema": schemas.get("output", {})
        }
    
    @classmethod
    def from_callable(
        cls,
        func: Callable,
        name: Optional[str] = None,
        docstring: Optional[str] = None,
        strict_schema: bool = True
    ) -> 'CallableTool':
        """Creates a new tool from a callable function."""
        func_name = name or func.__name__
        
        # Let the validators handle registration
        return cls(
            name=func_name,
            docstring=docstring or func.__doc__,
            strict_schema=strict_schema
        )

    @classmethod
    def from_registry(cls, name: str) -> 'CallableTool':
        """Creates a new tool from an existing registry entry."""
        ca_registry = CallableRegistry()
        if not ca_registry.get(name):
            raise ValueError(f"No callable found in registry with name: {name}")
            
        # Let the validators handle registration
        return cls(name=name)
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the callable with the given input data."""
        ca_registry = CallableRegistry()
        ca_registry._logger.debug(f"CallableTool({self.id}): Executing '{self.name}'")
        
        try:
            result = ca_registry.execute(self.name, input_data)
            ca_registry._logger.debug(f"CallableTool({self.id}): Execution successful")
            return result
        except Exception as e:
            ca_registry._logger.error(f"CallableTool({self.id}): Execution failed for '{self.name}'")
            raise ValueError(f"Error executing {self.name}: {str(e)}") from e
    
    async def aexecute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the callable with the given input data asynchronously.
        
        Will execute async functions natively and wrap sync functions in
        asyncio.to_thread automatically.
        
        Args:
            input_data: Input data for the callable
            
        Returns:
            Execution results
            
        Raises:
            ValueError: If execution fails
        """
        ca_registry = CallableRegistry()
        ca_registry._logger.debug(f"CallableTool({self.id}): Executing '{self.name}' asynchronously")
        
        try:
            result = await ca_registry.aexecute(self.name, input_data)
            ca_registry._logger.debug(f"CallableTool({self.id}): Async execution successful")
            return result
        except Exception as e:
            ca_registry._logger.error(f"CallableTool({self.id}): Async execution failed for '{self.name}'")
            raise ValueError(f"Error executing {self.name} asynchronously: {str(e)}") from e