"""
Entity models with registry integration and serialization support.

This module provides:
1. Base Entity class for registry-integrated, serializable objects
2. CallableTool implementation for managing executable functions
3. Serialization and persistence capabilities
4. Registry integration for both entities and callables
"""
from typing import Dict, Any, Optional, ClassVar, Type, TypeVar, List, Generic, Callable, Literal, Union, Tuple
from enum import Enum

from uuid import UUID, uuid4
from pydantic import BaseModel, Field, model_validator, computed_field
from pathlib import Path
import json
from datetime import datetime
import inspect

from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from anthropic.types import ToolParam, CacheControlEphemeralParam
from minference.utils import msg_dict_to_oai, msg_dict_to_anthropic, parse_json_string

from minference.lite.enregistry import EntityRegistry
from minference.lite.caregistry import (
    CallableRegistry,
    derive_input_schema,
    derive_output_schema,
    validate_schema_compatibility,
)
from openai.types.shared_params.response_format_json_schema import ResponseFormatJSONSchema, JSONSchema

from openai.types.shared_params import (
    ResponseFormatText,
    ResponseFormatJSONObject,
    FunctionDefinition
)
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletion
)
from anthropic.types import (
    MessageParam,
    TextBlock,
    ToolUseBlock,
    ToolParam,
    TextBlockParam,
    Message as AnthropicMessage
)
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
    
    def get_openai_tool(self) -> Optional[ChatCompletionToolParam]:
        """Get OpenAI tool format using the callable's schema."""
        if self.input_schema:
            return ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=self.name,
                    description=self.docstring or f"Execute {self.name} function",
                    parameters=self.input_schema
                )
            )
        return None

    def get_anthropic_tool(self) -> Optional[ToolParam]:
        """Get Anthropic tool format using the callable's schema."""
        if self.input_schema:
            return ToolParam(
                name=self.name,
                description=self.docstring or f"Execute {self.name} function",
                input_schema=self.input_schema,
                cache_control=CacheControlEphemeralParam(type='ephemeral')
            )
        return None
        
class StructuredTool(Entity):
    """
    Entity representing a tool for structured output with schema validation and LLM integration.
    
    Inherits from Entity for registry integration and automatic registration.
    Provides schema validation and LLM format conversion for structured outputs.
    """
    name: str = Field(
        default="generate_structured_output",
        description="Name for the structured output schema"
    )
    
    description: str = Field(
        default="Generate a structured output based on the provided JSON schema.",
        description="Description of what the structured output represents"
    )
    
    instruction_string: str = Field(
        default="Please follow this JSON schema for your response:",
        description="Instruction to prepend to schema for LLM"
    )
    
    json_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema defining the expected structure"
    )
    
    strict_schema: bool = Field(
        default=True,
        description="Whether to enforce strict schema validation"
    )

    def _custom_serialize(self) -> Dict[str, Any]:
        """Serialize tool-specific data."""
        return {
            "json_schema": self.json_schema,
            "description": self.description,
            "instruction": self.instruction_string
        }
    
    @classmethod
    def _custom_deserialize(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize tool-specific data."""
        return {
            "json_schema": data.get("json_schema", {}),
            "description": data.get("description", ""),
            "instruction_string": data.get("instruction", "")
        }

    @property
    def schema_instruction(self) -> str:
        """Get formatted schema instruction for LLM."""
        return f"{self.instruction_string}: {self.json_schema}"

    def get_openai_tool(self) -> Optional[ChatCompletionToolParam]:
        """Get OpenAI tool format."""
        if self.json_schema:
            return ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=self.name,
                    description=self.description,
                    parameters=self.json_schema
                )
            )
        return None

    def get_anthropic_tool(self) -> Optional[ToolParam]:
        """Get Anthropic tool format."""
        if self.json_schema:
            return ToolParam(
                name=self.name,
                description=self.description,
                input_schema=self.json_schema,
                cache_control=CacheControlEphemeralParam(type='ephemeral')
            )
        return None

    @classmethod
    def from_pydantic(
        cls,
        model: Type[BaseModel],
        name: Optional[str] = None,
        description: Optional[str] = None,
        instruction_string: Optional[str] = None,
        strict_schema: bool = True
    ) -> 'StructuredTool':
        """
        Create a StructuredTool from a Pydantic model.
        
        Args:
            model: Pydantic model class
            name: Optional override for tool name (defaults to model name)
            description: Optional override for description (defaults to model docstring)
            instruction_string: Optional custom instruction
            strict_schema: Whether to enforce strict schema validation
        """
        if not issubclass(model, BaseModel):
            raise ValueError("Model must be a Pydantic model")
            
        # Get model schema
        schema = model.model_json_schema()
        
        # Use model name if not provided
        tool_name = name or model.__name__.lower()
        
        # Use model docstring if no description provided
        tool_description = description or model.__doc__ or f"Generate {tool_name} structured output"
        
        return cls(
            name=tool_name,
            json_schema=schema,
            description=tool_description,
            instruction_string=instruction_string or cls.model_fields["instruction_string"].default,
            strict_schema=strict_schema
        )

class LLMClient(str, Enum):
    openai = "openai"
    azure_openai = "azure_openai"
    anthropic = "anthropic"
    vllm = "vllm"
    litellm = "litellm"

class ResponseFormat(str, Enum):
    json_beg = "json_beg"
    text = "text"
    json_object = "json_object"
    structured_output = "structured_output"
    tool = "tool"
    auto_tools = "auto_tools"
    workflow = "workflow"
    
class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    tool = "tool"
    system = "system"

class LLMConfig(Entity):
    """
    Configuration entity for LLM interactions.
    
    Specifies the client, model, and response format settings
    for interacting with various LLM providers.
    """
    client: LLMClient = Field(
        description="The LLM client/provider to use"
    )
    
    model: Optional[str] = Field(
        default=None,
        description="Model identifier for the LLM"
    )
    
    max_tokens: int = Field(
        default=400,
        description="Maximum number of tokens in completion",
        ge=1
    )
    
    temperature: float = Field(
        default=0,
        description="Sampling temperature",
        ge=0,
        le=2
    )
    
    response_format: ResponseFormat = Field(
        default=ResponseFormat.text,
        description="Format for LLM responses"
    )
    
    use_cache: bool = Field(
        default=True,
        description="Whether to use response caching"
    )

    @model_validator(mode="after")
    def validate_response_format(self) -> 'LLMConfig':
        """Validate response format compatibility with selected client."""
        if (self.response_format == ResponseFormat.json_object and 
            self.client in [LLMClient.vllm, LLMClient.litellm, LLMClient.anthropic]):
            raise ValueError(f"{self.client} does not support json_object response format")
            
        if (self.response_format == ResponseFormat.structured_output and 
            self.client == LLMClient.anthropic):
            raise ValueError(
                f"Anthropic does not support structured_output response format. "
                "Use json_beg or tool instead"
            )
            
        return self


ToolType = Literal["Callable", "Structured"]

class ChatMessage(Entity):
    """A chat message entity using chatml format."""
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the message was created"
    )
    
    role: MessageRole = Field(
        description="Role of the message sender" 
    )
    
    content: str = Field(
        description="The message content"
    )

    author_uuid: Optional[UUID] = Field(
        default=None,
        description="UUID of the author of the message"
    )

    chat_thread_uuid: Optional[UUID] = Field(
        default=None,
        description="UUID of the chat thread this message belongs to"
    )
    
    parent_message_uuid: Optional[UUID] = Field(
        default=None,
        description="UUID of the parent message in the conversation"
    )
    
    tool_name: Optional[str] = Field(
        default=None, 
        description="Name of the tool if this is a tool-related message"
    )

    tool_uuid: Optional[UUID] = Field(
        default=None,
        description="UUID of the tool in our EntityRegistry"
    )

    tool_type: Optional[ToolType] = Field(
        default=None,
        description="Type of tool - either Callable or Structured"
    )
    
    oai_tool_call_id: Optional[str] = Field(
        default=None,
        description="OAI tool call id"
    )
    
    tool_json_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON schema for tool input/output"
    )
    
    tool_call: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tool call details and arguments"
    )

    @property
    @computed_field
    def is_root(self) -> bool:
        """Check if this is a root message (no parent)."""
        return self.parent_message_uuid is None

    def get_parent(self) -> Optional['ChatMessage']:
        """Get the parent message if it exists."""
        if self.parent_message_uuid:
            return ChatMessage.get(self.parent_message_uuid)
        return None

    def get_tool(self) -> Optional[Union['CallableTool', 'StructuredTool']]:
        """Get the associated tool from the registry if it exists."""
        if not self.tool_uuid:
            return None
        
        if self.tool_type == "Callable":
            return CallableTool.get(self.tool_uuid)
        elif self.tool_type == "Structured":
            return StructuredTool.get(self.tool_uuid)
        return None
    

    def to_dict(self) -> Dict[str, Any]:
        """Convert to chatml format dictionary."""
        if self.role == MessageRole.tool:
            return {
                "role": self.role.value,
                "content": self.content,
                "tool_call_id": self.oai_tool_call_id
            }
        elif self.role == MessageRole.assistant and self.oai_tool_call_id is not None:
            return {
                "role": self.role.value,
                "content": self.content,
                "tool_calls": [{
                    "id": self.oai_tool_call_id,
                    "function": {
                        "arguments": json.dumps(self.tool_call),
                        "name": self.tool_name
                    },
                    "type": "function"
                }]
            }
        else:
            return {
                "role": self.role.value,
                "content": self.content
            }

    @classmethod
    def from_dict(cls, message_dict: Dict[str, Any]) -> 'ChatMessage':
        """Create a ChatMessage from a chatml format dictionary."""
        return cls(
            role=MessageRole(message_dict["role"]),
            content=message_dict["content"]
        )

class SystemPrompt(Entity):
    """Entity representing a reusable system prompt."""
    
    name: str = Field(
        description="Identifier name for the system prompt"
    )
    
    content: str = Field(
        description="The system prompt text content"
    )

class Thread(Entity):
    """A chat thread entity managing conversation flow and message history."""
    
    name: Optional[str] = Field(
        default=None,
        description="Optional name for the thread"
    )
    
    system_prompt: Optional[SystemPrompt] = Field(
        default=None,
        description="Associated system prompt"
    )
    
    history: List[ChatMessage] = Field(
        default_factory=list,
        description="Messages in chronological order"
    )
    
    new_message: Optional[str] = Field(
        default=None,
        description="Temporary storage for message being processed"
    )
    
    prefill: str = Field(
        default="Here's the valid JSON object response:```json",
        description="Prefill assistant response with an instruction"
    )
    
    postfill: str = Field(
        default="\n\nPlease provide your response in JSON format.",
        description="Postfill user response with an instruction"
    )
    
    use_schema_instruction: bool = Field(
        default=False,
        description="Whether to use the schema instruction"
    )
    
    use_history: bool = Field(
        default=True,
        description="Whether to use the history"
    )
    
    structured_output: Optional[StructuredTool] = Field(
        default=None,
        description="Associated structured output tool"
    )
    
    llm_config: LLMConfig = Field(
        description="LLM configuration"
    )
    
    tools: List[Union[CallableTool, StructuredTool]] = Field(
        default_factory=list,
        description="Available tools"
    )

    @property
    def oai_response_format(self) -> Optional[Union[ResponseFormatText, ResponseFormatJSONObject, ResponseFormatJSONSchema]]:
        """Get OpenAI response format based on config."""
        if self.llm_config.response_format == ResponseFormat.text:
            return ResponseFormatText(type="text")
        elif self.llm_config.response_format == ResponseFormat.json_object:
            return ResponseFormatJSONObject(type="json_object")
        elif self.llm_config.response_format == ResponseFormat.structured_output:
            assert self.structured_output is not None, "Structured output is not set"
            return self.structured_output.get_openai_json_schema_response()
        return None

    @property
    def use_prefill(self) -> bool:
        """Check if prefill should be used."""
        return (self.llm_config.client in [LLMClient.anthropic, LLMClient.vllm, LLMClient.litellm] and 
                self.llm_config.response_format == ResponseFormat.json_beg)

    @property
    def use_postfill(self) -> bool:
        """Check if postfill should be used."""
        return (self.llm_config.client == LLMClient.openai and 
                self.llm_config.response_format in [ResponseFormat.json_object, ResponseFormat.json_beg] and 
                not self.use_schema_instruction)

    @property
    def system_message(self) -> Optional[Dict[str, str]]:
        """Get system message including schema instruction if needed."""
        content = self.system_prompt.content if self.system_prompt else ""
        if self.use_schema_instruction and self.structured_output:
            content = "\n".join([content, self.structured_output.schema_instruction])
        return {"role": "system", "content": content} if content else None

    @property
    def message_objects(self) -> List[ChatMessage]:
        """Get all message objects in the conversation."""
        messages = []
        
        # Add system message
        if self.system_message:
            messages.append(ChatMessage(
                role=MessageRole.system,
                content=self.system_message["content"]
            ))
            
        # Add history
        if self.use_history:
            messages.extend(self.history)
            
        # Add new message
        if self.new_message:
            messages.append(ChatMessage(
                role=MessageRole.user,
                content=self.new_message
            ))
            
        # Handle prefill/postfill
        if self.use_prefill and messages:
            messages.append(ChatMessage(
                role=MessageRole.assistant,
                content=self.prefill
            ))
        elif self.use_postfill and messages:
            messages[-1].content += self.postfill
            
        return messages

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Get messages in chatml dict format."""
        return [msg.to_dict() for msg in self.message_objects]

    @property
    def oai_messages(self) -> List[ChatCompletionMessageParam]:
        """Get messages in OpenAI format."""
        return msg_dict_to_oai(self.messages)

    @property
    def anthropic_messages(self) -> Tuple[List[TextBlockParam], List[MessageParam]]:
        """Get messages in Anthropic format."""
        return msg_dict_to_anthropic(self.messages, use_cache=self.llm_config.use_cache)

    @property
    def vllm_messages(self) -> List[ChatCompletionMessageParam]:
        """Get messages in vLLM format."""
        return msg_dict_to_oai(self.messages)

    def get_tool_by_name(self, tool_name: str) -> Optional[Union[CallableTool, StructuredTool]]:
        """Get tool by name from available tools."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        if self.structured_output and self.structured_output.name == tool_name:
            return self.structured_output
        return None

    def add_user_message(self) -> ChatMessage:
        """Add current new_message as user message."""
        if self.new_message is None:
            raise ValueError("new_message is None, cannot add to history")

        last_message = self.history[-1] if self.history else None
        user_message = ChatMessage(
            role=MessageRole.user,
            content=self.new_message,
            parent_message_uuid=last_message.id if last_message else None
        )
        self.history.append(user_message)
        self.new_message = None
        return user_message

    def add_assistant_response(self, content: str, tool_name: Optional[str] = None,
                             tool_call: Optional[Dict[str, Any]] = None,
                             parent_message_uuid: Optional[UUID] = None) -> ChatMessage:
        """Add assistant response to history."""
        message = ChatMessage(
            role=MessageRole.assistant,
            content=content,
            parent_message_uuid=parent_message_uuid,
            tool_name=tool_name
        )
        if tool_call:
            message.tool_call = tool_call
            if tool := self.get_tool_by_name(tool_name):
                message.tool_uuid = tool.id
                message.tool_json_schema = tool.input_schema if isinstance(tool, CallableTool) else tool.json_schema
                message.tool_type = "Callable" if isinstance(tool, CallableTool) else "Structured"

        self.history.append(message)
        return message

    def get_tools_for_llm(self) -> Optional[List[Union[ChatCompletionToolParam, ToolParam]]]:
        """Get tools in format appropriate for current LLM."""
        if not self.tools:
            return None
            
        tools = []
        for tool in self.tools:
            if self.llm_config.client in [LLMClient.openai, LLMClient.vllm, LLMClient.litellm]:
                tools.append(tool.get_openai_tool())
            elif self.llm_config.client == LLMClient.anthropic:
                tools.append(tool.get_anthropic_tool())
        return tools if tools else None