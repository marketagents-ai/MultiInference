"""
Entity models with registry integration and serialization support.

This module provides:
1. Base Entity class for registry-integrated, serializable objects
2. CallableTool implementation for managing executable functions
3. Serialization and persistence capabilities
4. Registry integration for both entities and callables
"""
from typing import Dict, Any, Optional, ClassVar, Type, TypeVar, List, Generic, Callable, Literal, Union, Tuple, Self, Annotated
from enum import Enum

from uuid import UUID, uuid4
from pydantic import BaseModel, Field, model_validator, computed_field, ValidationInfo, AfterValidator
from pathlib import Path
import json
from datetime import datetime
import inspect
from jsonschema import validate
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from anthropic.types import ToolParam, CacheControlEphemeralParam
from minference.clients.utils import msg_dict_to_oai, msg_dict_to_anthropic, parse_json_string

from minference.ecs.caregistry import (
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
    ToolResultBlockParam,
    Message as AnthropicMessage
)

from typing import List, Optional, Dict, Any, Union, Tuple, Self, TypeVar, Type, Generic, Callable, Literal
from typing_extensions import Literal
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice, ChatCompletion
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion import Choice
from logging import getLogger, WARNING, INFO

from minference.ecs.entity import Entity, EntityRegistry, entity_tracer

T_Self = TypeVar('T_Self', bound='CallableTool')
logger = getLogger("minference.threads.models")
logger.setLevel(WARNING)


class CallableTool(Entity):
    """
    An immutable callable tool that can be registered and executed with schema validation.
    
    Inherits from Entity for registry integration and serialization support.
    The tool is registered in:
    - CallableRegistry: For the actual function registration and execution
    - EntityRegistry: For the tool metadata and versioning (handled by parent)
    
    Any modifications require creating new instances with new UUIDs.
    """    
    name: Annotated[str, AfterValidator(lambda x: x)] = Field(
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
            ],
            "populate_by_name": True
        }
    }
    
    @classmethod
    def _get_or_register_function(
        cls,
        func: Optional[Callable] = None,
        name: Optional[str] = None,
        callable_text: Optional[str] = None
    ) -> tuple[str, Callable]:
        """
        Helper method to get an existing function from registry or register a new one.
        
        Args:
            func: Optional callable to register
            name: Optional name for the function
            callable_text: Optional source code for the function
            
        Returns:
            Tuple of (function_name, callable_function)
            
        Raises:
            ValueError: If insufficient information provided or registration fails
        """
        registry = CallableRegistry
        
        # Determine function name
        if name is None:
            if func is not None:
                name = func.__name__
            else:
                raise ValueError("Must provide either a name or a callable function")
                
        # Check if function already exists in registry
        existing_func = registry.get(name)
        if existing_func is not None:
            return name, existing_func
            
        # Register new function
        if func is not None:
            registry.register(name, func)
            return name, func
        elif callable_text is not None:
            registry.register_from_text(name, callable_text)
            registered_func = registry.get(name)
            if registered_func is None:
                raise ValueError(f"Failed to register function '{name}' from source code")
            return name, registered_func
        else:
            raise ValueError(
                f"No function found in registry for '{name}' and no function or "
                "callable_text provided for registration"
            )

    @classmethod
    def from_callable(cls, func: Callable, name: Optional[str] = None, docstring: Optional[str] = None, strict_schema: bool = True) -> 'CallableTool':
        """Creates a new tool from a callable function."""
        # Get function source if possible
        try:
            callable_text = inspect.getsource(func)
        except (TypeError, OSError):
            callable_text = str(func)
            
        # Get or register the function
        func_name, registered_func = cls._get_or_register_function(
            func=func,
            name=name,
            callable_text=callable_text
        )
        
        # Derive schemas
        input_schema = derive_input_schema(registered_func)
        output_schema = derive_output_schema(registered_func)
        
        # Create tool instance
        return cls(
            name=func_name,
            docstring=docstring or func.__doc__,
            input_schema=input_schema,
            output_schema=output_schema,
            strict_schema=strict_schema,
            callable_text=callable_text
        )

    @classmethod
    def from_registry(cls, name: str) -> 'CallableTool':
        """Creates a new tool from an existing registry entry."""
        # Get function from registry
        func_name, registered_func = cls._get_or_register_function(name=name)
        
        # Get function source if possible
        try:
            callable_text = inspect.getsource(registered_func)
        except (TypeError, OSError):
            callable_text = str(registered_func)
        
        # Create tool instance
        return cls(
            name=func_name,
            docstring=registered_func.__doc__,
            callable_text=callable_text
        )

    @classmethod
    def from_source(cls, source: str, name: Optional[str] = None, docstring: Optional[str] = None, strict_schema: bool = True) -> 'CallableTool':
        """Creates a new tool from source code string."""
        # Get or register the function
        func_name, registered_func = cls._get_or_register_function(
            callable_text=source,
            name=name
        )
        
        # Create tool instance
        return cls(
            name=func_name,
            docstring=docstring or registered_func.__doc__,
            strict_schema=strict_schema,
            callable_text=source
        )

    @model_validator(mode='before')
    def validate_schemas_and_callable(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validates the tool's schemas and ensures its callable is registered. Must run before entity registration."""
        # Ensure we have a valid name
        name = data.get('name')
        if not isinstance(name, str):
            raise ValueError("Tool name must be a string")
            
        # Get or register the function
        func_name, func = cls._get_or_register_function(
            name=name,
            callable_text=data.get('callable_text')
        )
        
        # Store text representation if not provided
        if not data.get('callable_text'):
            try:
                data['callable_text'] = inspect.getsource(func)
            except (TypeError, OSError):
                data['callable_text'] = str(func)
        
        # Check if function is already registered
        registry = CallableRegistry
        existing_info = registry.get_info(name)
        
        if existing_info:
            # Function exists, preserve its schemas
            if not data.get('input_schema'):
                data['input_schema'] = existing_info.input_schema
            if not data.get('output_schema'):
                data['output_schema'] = existing_info.output_schema
        else:
            # New function, derive and validate schemas
            derived_input = derive_input_schema(func)
            derived_output = derive_output_schema(func)
            
            # Validate and set input schema
            if data.get('input_schema'):
                validate_schema_compatibility(derived_input, data['input_schema'])
            else:
                data['input_schema'] = derived_input
                
            # Validate and set output schema
            if data.get('output_schema'):
                validate_schema_compatibility(derived_output, data['output_schema'])
            else:
                data['output_schema'] = derived_output
            
        return data
    
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
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the callable with the given input data."""
        ca_registry = CallableRegistry
        ca_registry._logger.info(f"CallableTool({self.id}): Executing '{self.name}'")
        
        try:
            result = ca_registry.execute(self.name, input_data)
            ca_registry._logger.info(f"CallableTool({self.id}): Execution successful")
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
        ca_registry = CallableRegistry
        ca_registry._logger.info(f"CallableTool({self.id}): Executing '{self.name}' asynchronously")
        
        try:
            result = await ca_registry.aexecute(self.name, input_data)
            ca_registry._logger.info(f"CallableTool({self.id}): Async execution successful")
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

    def get_anthropic_tool(self, use_cache:bool=True) -> Optional[ToolParam]:
        """Get Anthropic tool format using the callable's schema."""
        if self.input_schema:
            if use_cache:
                return ToolParam(
                    name=self.name,
                    description=self.docstring or f"Execute {self.name} function",
                    input_schema=self.input_schema,
                    cache_control=CacheControlEphemeralParam(type='ephemeral')
                )
            else:
                return ToolParam(
                    name=self.name,
                    description=self.docstring or f"Execute {self.name} function",
                    input_schema=self.input_schema,
            )
        return None
        
    def fork(self: T_Self, **kwargs: Any) -> T_Self:
        """
        Override fork to preserve schemas when creating new versions.
        """
        # Preserve schemas if not explicitly changed
        if 'input_schema' not in kwargs:
            kwargs['input_schema'] = self.input_schema
        if 'output_schema' not in kwargs:
            kwargs['output_schema'] = self.output_schema
            
        # Call parent fork with preserved schemas
        return super().fork(**kwargs)

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

    post_validate_schema: bool = Field(
        default=True,
        description="Whether to post-validate the schema"
    )

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute structured output validation."""
        
        if self.post_validate_schema:
            logger.info(f"StructuredTool({self.id}): Validating input for '{self.name}'")
            try:
                # Validate input against schema
                validate(instance=input_data, schema=self.json_schema)
                logger.info(f"StructuredTool({self.id}): Validation successful")
                return input_data
            
            except Exception as e:
                    logger.error(f"StructuredTool({self.id}): Validation failed - {str(e)}")
                    return {"error": str(e)}
        else:
            logger.info(f"StructuredTool({self.id}): Validation skipped for '{self.name}'")
            return input_data

    async def aexecute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async version of execute - for consistency with CallableTool interface.
        Since validation is CPU-bound, we don't need true async here.
        """
        return self.execute(input_data)

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

    def get_anthropic_tool(self,use_cache:bool=True) -> Optional[ToolParam]:
        """Get Anthropic tool format."""
        if self.json_schema:
            if use_cache:
                return ToolParam(
                    name=self.name,
                    description=self.description,
                    input_schema=self.json_schema,
                    cache_control=CacheControlEphemeralParam(type='ephemeral')
                )
            else:
                return ToolParam(
                    name=self.name,
                    description=self.description,
                    input_schema=self.json_schema,
            )
        return None

    def get_openai_json_schema_response(self) -> Optional[ResponseFormatJSONSchema]:
        """Get OpenAI JSON schema response format."""
        if self.json_schema:
            schema = JSONSchema(
                name=self.name,
                description=self.description,
                schema=self.json_schema,
                strict=self.strict_schema
            )
            return ResponseFormatJSONSchema(type="json_schema", json_schema=schema)
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

    @model_validator(mode='before')
    def validate_history(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-validate to ensure history contains proper ChatMessage objects"""
        logger.debug(f"Starting history validation for ChatThread")
        
        if 'history' in values and isinstance(values['history'], list):
            logger.info(f"Processing history list with {len(values['history'])} messages")
            history = []
            for idx, msg in enumerate(values['history']):
                logger.debug(f"Processing message {idx}: {type(msg)}")
                if isinstance(msg, dict):
                    logger.debug(f"Converting dict to ChatMessage: {msg}")
                    msg = ChatMessage.model_validate(msg)
                history.append(msg)
            logger.info(f"History validation complete - processed {len(history)} messages")
            values['history'] = history
        else:
            logger.debug("No history found or history is not a list")
        return values

class LLMClient(str, Enum):
    openai = "openai"
    azure_openai = "azure_openai"
    anthropic = "anthropic"
    vllm = "vllm"
    litellm = "litellm"
    openrouter = "openrouter"

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

    reasoner: bool = Field(
        default=False,
        description="Whether to use reasoning"
    )

    reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="The effort level for reasoning"
    )

    @model_validator(mode="after")
    def validate_response_format(self) -> Self:
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
        
        if self.reasoner and self.client != LLMClient.openai:
            raise ValueError("Reasoning response format is only supported for OpenAI 01 and 03")
            
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

    chat_thread_id: Optional[UUID] = Field(
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

    usage: Optional['Usage'] = Field(
        default=None,
        description="Usage statistics for the message"
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



class Usage(Entity):
    """Tracks token usage for LLM interactions."""
    model: str = Field(
        description="The model used for the interaction"
    )
    prompt_tokens: int = Field(
        description="Number of tokens in the prompt"
    )
    completion_tokens: int = Field(
        description="Number of tokens in the completion"
    )
    total_tokens: int = Field(
        description="Total number of tokens used"
    )
    cache_creation_input_tokens: Optional[int] = Field(
        default=None,
        description="Number of tokens used in cache creation with Anthropic endpoint"
    )
    cache_read_input_tokens: Optional[int] = Field(
        default=None,
        description="Number of tokens read from cache with Anthropic endpoint"
    )
    accepted_prediction_tokens: Optional[int] = Field(
        default=None,
        description="Number of accepted prediction tokens using OpenAI endpoint"
    )   
    audio_tokens: Optional[int] = Field(
        default=None,
        description="Number of audio tokens using OpenAI endpoint"
    )
    reasoning_tokens: Optional[int] = Field(
        default=None,
        description="Number of reasoning tokens using OpenAI endpoint"
    )
    rejected_prediction_tokens: Optional[int] = Field(
        default=None,
        description="Number of rejected prediction tokens using OpenAI endpoint"
    )
    cached_tokens: Optional[int] = Field(
        default=None,
        description="Number of tokens read from cache with OpenAI endpoint"
    )


class GeneratedJsonObject(Entity):
    """Represents a structured JSON object generated by an LLM."""
    name: str = Field(
        description="Name identifier for the generated object"
    )
    object: Dict[str, Any] = Field(
        description="The actual JSON object content"
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        description="Associated tool call ID if generated via tool"
    )




class DeepSeekChatCompletionMessage(ChatCompletionMessage):
    """Extended ChatCompletionMessage to include DeepSeek-specific fields."""
    reasoning: Optional[str] = None
    """DeepSeek-specific reasoning field explaining the model's thought process."""

class DeepSeekChoice(Choice):
    """Extended Choice to use DeepSeekChatCompletionMessage."""
    message: DeepSeekChatCompletionMessage
    """A chat completion message generated by the model, with DeepSeek extensions."""

class DeepSeekChatCompletion(ChatCompletion):
    """Extended ChatCompletion model to handle DeepSeek-specific fields."""
    id: str
    """A unique identifier for the chat completion."""

    choices: List[DeepSeekChoice]
    """A list of chat completion choices with DeepSeek extensions."""

    created: int
    """The Unix timestamp (in seconds) of when the chat completion was created."""

    model: str
    """The model used for the chat completion."""

    object: Literal["chat.completion"]
    """The object type, which is always `chat.completion`."""

    usage: Optional[CompletionUsage] = None
    """Usage statistics for the completion request."""

class RawOutput(Entity):
    """Raw output from LLM API calls with metadata."""
    raw_result: Dict[str, Any] = Field(
        description="Raw response from the LLM API"
    )
    completion_kwargs: Dict[str, Any] = Field(
        description="Arguments used in the completion call"
    )
    start_time: float = Field(
        description="Timestamp when the API call started"
    )
    end_time: float = Field(
        description="Timestamp when the API call completed"
    )
    chat_thread_id: Optional[UUID] = Field(
        default=None,
        description="ID of the associated chat thread, can be used to retrieve from registry a frozen copy of the chat thread at creation time"
    )
    chat_thread_live_id: Optional[UUID] = Field(
        default=None,
        description="Live ID of the associated chat thread"
    )
    client: LLMClient = Field(
        description="The LLM client used for this call"
    )
    parsed_result: Optional[Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage], Optional[str]]] = Field(
        default=None,
        description="Cached parsed results (content, json_object, usage, error)",
        exclude=True
    )

    @property
    def time_taken(self) -> float:
        """Calculate time taken for the LLM call."""
        return self.end_time - self.start_time

    @computed_field
    @property
    def str_content(self) -> Optional[str]:
        """Extract string content from raw result."""
        return self._parse_result()[0]

    @computed_field
    @property
    def json_object(self) -> Optional[GeneratedJsonObject]:
        """Extract JSON object from raw result."""
        return self._parse_result()[1]
    
    @computed_field
    @property
    def usage(self) -> Optional[Usage]:
        """Extract usage statistics."""
        return self._parse_result()[2]

    @computed_field
    @property
    def error(self) -> Optional[str]:
        """Extract error message if present."""
        return self._parse_result()[3]

    @computed_field
    @property
    def contains_object(self) -> bool:
        """Check if result contains a JSON object."""
        return self._parse_result()[1] is not None

    def _parse_result(self) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage], Optional[str]]:
        """Parse raw result into structured components with caching."""
        if self.parsed_result is not None:
            return self.parsed_result
            
        # Check for errors first
        if getattr(self.raw_result, "error", None):
            self.parsed_result = (None, None, None, getattr(self.raw_result, "error", None))
            return self.parsed_result

        provider = self.result_provider
        try:
            # Try DeepSeek format first for OpenRouter responses
            if provider == LLMClient.openrouter:
                try:
                    result = DeepSeekChatCompletion.model_validate(self.raw_result)
                    self.parsed_result = self.parse_oai_completion(result)
                except:
                    # Fall back to standard OpenAI format
                    result = ChatCompletion.model_validate(self.raw_result)
                    self.parsed_result = self.parse_oai_completion(result)
            # Handle other providers as before
            elif provider == LLMClient.openai:
                self.parsed_result = self.parse_oai_completion(ChatCompletion.model_validate(self.raw_result))
            elif provider == LLMClient.anthropic:
                self.parsed_result = self.parse_anthropic_message(AnthropicMessage.model_validate(self.raw_result))
            elif provider in [LLMClient.vllm, LLMClient.litellm]:
                self.parsed_result = self.parse_oai_completion(ChatCompletion.model_validate(self.raw_result))
            else:
                raise ValueError(f"Unsupported result provider: {provider}")
        except Exception as e:
            self.parsed_result = (None, None, None, str(e))
            
        return self.parsed_result if self.parsed_result is not None else (None, None, None, None)

    @computed_field
    @property
    def result_provider(self) -> Optional[LLMClient]:
        """Determine the LLM provider from the result format."""
        return self.search_result_provider() if self.client is None else self.client
    
    def search_result_provider(self) -> Optional[LLMClient]:
        """Identify LLM provider from result structure."""
        try:
            ChatCompletion.model_validate(self.raw_result)
            # Check if it's specifically OpenRouter
            if "openrouter" in getattr(self.raw_result, "model", "").lower():
                return LLMClient.openrouter
            return LLMClient.openai
        except:
            try:
                AnthropicMessage.model_validate(self.raw_result)
                return LLMClient.anthropic
            except:
                return None

    def _parse_json_string(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON string safely."""
        try:
            cleaned_content = content
            import re
            
            # Try stripping tool_call tags first
            tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
            if '<tool_call>' in cleaned_content:
                match = re.search(tool_call_pattern, cleaned_content, re.DOTALL)
                if match:
                    cleaned_content = match.group(1).strip()
            
            # Try stripping tool_request tags
            tool_request_pattern = r'\[TOOL_REQUEST\](.*?)\[END_TOOL_REQUEST\]'
            if '[TOOL_REQUEST]' in cleaned_content:
                match = re.search(tool_request_pattern, cleaned_content, re.DOTALL)
                if match:
                    cleaned_content = match.group(1).strip()

            # Try direct JSON parsing of cleaned content
            try:
                return json.loads(cleaned_content)
            except json.JSONDecodeError:
                pass

            # Try original content if cleaned parsing failed
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass

            # Try code block format
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            # Try Python-style boolean conversion
            try:
                cleaned_content = re.sub(r'\bTrue\b', 'true', cleaned_content)
                cleaned_content = re.sub(r'\bFalse\b', 'false', cleaned_content)
                cleaned_content = re.sub(r'\bNone\b', 'null', cleaned_content)
                return json.loads(cleaned_content)
            except json.JSONDecodeError:
                pass

            return None
            
        except Exception as e:
            logger.error(f"Error parsing JSON string: {str(e)}")
            return None

    def parse_oai_completion(self, chat_completion: Union[ChatCompletion, DeepSeekChatCompletion]) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage], Optional[str]]:
        """Parse OpenAI or DeepSeek completion format."""
        logger.info("Starting parse_oai_completion")
        
        # Extract the choice and message
        try:
            choice = chat_completion.choices[0]
            message = choice.message
        except (AttributeError, IndexError) as e:
            logger.error(f"Failed to extract message from completion: {e}")
            return None, None, None, None
        
        # Handle DeepSeek specific formatting
        if isinstance(chat_completion, DeepSeekChatCompletion) and isinstance(message, DeepSeekChatCompletionMessage):
            content = f"<think>{message.reasoning}</think>\n{message.content}" if message.reasoning else message.content
        else:
            content = message.content
        
        # Log initial content (if present)
        if content:
            logger.info(f"Initial content: {content[:200]}...")
        else:
            logger.info("Content is None or empty")
        
        json_object = None
        usage = None
        
        # Part 1: Handle explicit tool calls (OpenAI format)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            json_object = self._parse_explicit_tool_calls(message.tool_calls)
        
        # Part 2: Handle content-based tool calls (for vLLM and others)
        elif content:
            json_object = self._parse_content_tool_calls(content)
            
            # If a tool call was found in the content, set content to None
            if json_object:
                content = None
        
        # Part 3: Extract usage information
        if hasattr(chat_completion, 'usage'):
            usage = self._extract_usage_info(chat_completion)
        
        logger.info(f"Parsing complete. Content: {'Present' if content else 'None'}, JSON object: {'Present' if json_object else 'None'}")
        return content, json_object, usage, None

    def _parse_explicit_tool_calls(self, tool_calls) -> Optional[GeneratedJsonObject]:
        """Parse explicit tool calls from OpenAI format."""
        if not tool_calls or not isinstance(tool_calls, list) or len(tool_calls) == 0:
            return None
        
        # Process the first tool call
        tool_call = tool_calls[0]
        
        try:
            # Extract name and ID
            name = tool_call.function.name
            tool_call_id = tool_call.id
            arguments = tool_call.function.arguments
            
            logger.info(f"Processing tool call: {name} (ID: {tool_call_id})")
            logger.debug(f"Raw arguments: {arguments[:200]}...")
            
            # First try to clean any XML-style tool call wrapping
            cleaned_arguments = arguments
            if '<tool_call>' in arguments:
                import re
                match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', arguments, re.DOTALL)
                if match:
                    cleaned_arguments = match.group(1).strip()
                    logger.info("Stripped XML tool_call tags from arguments")
            
            # Try parsing the cleaned arguments
            try:
                object_dict = json.loads(cleaned_arguments)
                logger.info(f"Successfully parsed cleaned tool arguments for {name}")
                
                # If this was wrapped in tool_call tags, check if it has name/arguments structure
                if '<tool_call>' in arguments and 'name' in object_dict and 'arguments' in object_dict:
                    return GeneratedJsonObject(
                        name=object_dict["name"],
                        object=object_dict["arguments"],
                        tool_call_id=tool_call_id
                    )
                
                return GeneratedJsonObject(
                    name=name, 
                    object=object_dict,
                    tool_call_id=tool_call_id
                )
                
            except json.JSONDecodeError as e:
                # If JSON parsing fails, try to clean up Python style booleans
                logger.debug(f"Initial JSON parsing failed: {e}, trying with boolean conversion")
                try:
                    # Replace Python-style booleans with JSON-style booleans
                    cleaned_args = re.sub(r'\bTrue\b', 'true', cleaned_arguments)
                    cleaned_args = re.sub(r'\bFalse\b', 'false', cleaned_args)
                    cleaned_args = re.sub(r'\bNone\b', 'null', cleaned_args)
                    
                    object_dict = json.loads(cleaned_args)
                    logger.info(f"Successfully parsed cleaned tool arguments with boolean conversion for {name}")
                    
                    # Check again for nested structure after boolean conversion
                    if '<tool_call>' in arguments and 'name' in object_dict and 'arguments' in object_dict:
                        return GeneratedJsonObject(
                            name=object_dict["name"],
                            object=object_dict["arguments"],
                            tool_call_id=tool_call_id
                        )
                    
                    return GeneratedJsonObject(
                        name=name, 
                        object=object_dict,
                        tool_call_id=tool_call_id
                    )
                    
                except json.JSONDecodeError:
                    # If all parsing attempts fail, log the error
                    logger.error(f"Failed to parse tool arguments as JSON after all attempts")
                    return None
        
        except Exception as e:
            logger.error(f"Error processing tool call: {e}")
            return None

    def _parse_content_tool_calls(self, content: str) -> Optional[GeneratedJsonObject]:
        """Parse tool calls from content for providers that don't use explicit tool_calls."""
        import re
        
        # Define patterns for tool calls in content
        patterns = [
            r'<tool_call>\s*(.*?)\s*</tool_call>',        # Lowercase pattern
            r'<TOOL_CALL>\s*(.*?)\s*</TOOL_CALL>',        # Uppercase pattern
            r'\[TOOL_REQUEST\](.*?)\[END_TOOL_REQUEST\]', # Alternative format
            r'<tool_use>(.*?)</tool_use>'                 # Anthropic-style pattern
        ]
        
        # Check if content contains any tool call patterns
        has_pattern = False
        for pattern in patterns:
            tag = pattern.split(r'\s*')[0][1:]
            has_pattern = tag in content
            if has_pattern:
                break
                
        logger.info(f"Content contains tool call pattern: {has_pattern}")
        
        if has_pattern:
            # Try each pattern
            for pattern in patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    logger.info(f"Found match with pattern: {pattern}")
                    
                    try:
                        # Parse the tool call JSON
                        tool_data = json.loads(match.group(1).strip())
                        logger.info(f"Successfully parsed tool call data from content")
                        
                        # Determine name and arguments
                        name = tool_data.get("name")
                        arguments = tool_data.get("arguments")
                        
                        # Handle different JSON structures
                        if name and arguments:
                            # Standard format: {"name": "tool_name", "arguments": {...}}
                            return GeneratedJsonObject(
                                name=name,
                                object=arguments,
                                tool_call_id=str(uuid4())
                            )
                        elif "function" in tool_data:
                            # OpenAI-like format: {"function": {"name": "tool_name", "arguments": {...}}}
                            func = tool_data["function"]
                            return GeneratedJsonObject(
                                name=func.get("name", "unknown_tool"),
                                object=json.loads(func.get("arguments", "{}")),
                                tool_call_id=str(uuid4())
                            )
                        else:
                            # Fallback: assume the whole object is the arguments
                            logger.warning("Unclear tool call structure, using default mapping")
                            return GeneratedJsonObject(
                                name="unknown_tool",
                                object=tool_data,
                                tool_call_id=str(uuid4())
                            )
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse tool call JSON from content: {e}")
                        logger.debug(f"Raw matched content: {match.group(1).strip()}")
        
        # If no tool call found, try parsing content as regular JSON
        logger.info("No tool call pattern match, trying regular JSON parsing")
        
        # Get schema name if available
        name = None
        if hasattr(self, 'completion_kwargs') and self.completion_kwargs:
            try:
                response_format = self.completion_kwargs.get("response_format", {})
                if response_format and isinstance(response_format, dict):
                    json_schema = response_format.get("json_schema", {})
                    if json_schema and isinstance(json_schema, dict):
                        name = json_schema.get("name")
                        if name:
                            logger.info(f"Found schema name from completion_kwargs: {name}")
            except Exception as e:
                logger.warning(f"Error extracting schema name: {e}")
        
        # Try to parse content as JSON
        parsed_json = self._parse_json_string(content)
        if parsed_json:
            logger.info("Successfully parsed content as regular JSON")
            return GeneratedJsonObject(
                name="parsed_content" if name is None else name,
                object=parsed_json,
                tool_call_id=str(uuid4())
            )
        
        # If all parsing attempts fail, return None
        return None

    def _extract_usage_info(self, chat_completion) -> Optional[Usage]:
        """Extract usage information from the completion."""
        try:
            usage_data = chat_completion.usage
            
            # Basic fields that should always be present
            usage_kwargs = {
                "model": getattr(chat_completion, "model", "unknown"),
                "prompt_tokens": usage_data.prompt_tokens,
                "completion_tokens": usage_data.completion_tokens,
                "total_tokens": usage_data.total_tokens
            }
            
            # Handle optional nested fields
            completion_tokens_details = getattr(usage_data, 'completion_tokens_details', None)
            prompt_tokens_details = getattr(usage_data, 'prompt_tokens_details', None)
            
            # Add optional fields if they exist
            optional_fields = {
                "accepted_prediction_tokens": None,
                "audio_tokens": None,
                "reasoning_tokens": None,
                "rejected_prediction_tokens": None,
                "cached_tokens": None
            }
            
            # Check completion_tokens_details
            if completion_tokens_details:
                optional_fields["accepted_prediction_tokens"] = getattr(completion_tokens_details, 'accepted_prediction_tokens', None)
                optional_fields["audio_tokens"] = getattr(completion_tokens_details, 'audio_tokens', None)
                optional_fields["reasoning_tokens"] = getattr(completion_tokens_details, 'reasoning_tokens', None)
                optional_fields["rejected_prediction_tokens"] = getattr(completion_tokens_details, 'rejected_prediction_tokens', None)
            
            # Check prompt_tokens_details
            if prompt_tokens_details:
                optional_fields["cached_tokens"] = getattr(prompt_tokens_details, 'cached_tokens', None)
            
            # Add non-None optional fields to kwargs
            for key, value in optional_fields.items():
                if value is not None:
                    usage_kwargs[key] = value
            
            return Usage(**usage_kwargs)
            
        except Exception as e:
            logger.warning(f"Error extracting usage information: {e}")
            return None

    def parse_anthropic_message(self, message: AnthropicMessage) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage], Optional[str]]:
        """Parse Anthropic message format."""
        content = None
        json_object = None
        usage = None

        if message.content:
            text_block = next((block for block in message.content 
                        if isinstance(block, TextBlock) and block.text.strip()), None)
            tool_block = next((block for block in message.content 
                        if isinstance(block, ToolUseBlock)), None)
            # Check if it's a TextBlock
            if isinstance(text_block, TextBlock):
                content = text_block.text

                parsed_json = self._parse_json_string(content)
                if parsed_json:
                    json_object = GeneratedJsonObject(
                        name="parsed_content", 
                        object=parsed_json
                    )
                    content = content
            # Check if it's a ToolUseBlock
            if isinstance(tool_block, ToolUseBlock):
            
                tool_call_id = tool_block.id
                # Cast tool_use.input to Dict[str, Any]
                if isinstance(tool_block.input, dict):
                    json_object = GeneratedJsonObject(
                        name=tool_block.name,
                        object=tool_block.input,
                        tool_call_id=tool_call_id
                    )
                else:
                    # Handle non-dict input by wrapping it

                    json_object = GeneratedJsonObject(
                        name=tool_block.name,
                        object={"value": tool_block.input},
                        tool_call_id=tool_call_id
                    )


        if hasattr(message, 'usage'):
            usage = Usage(
                model=message.model,
                prompt_tokens=message.usage.input_tokens,
                completion_tokens=message.usage.output_tokens,
                total_tokens=message.usage.input_tokens + message.usage.output_tokens,
                cache_creation_input_tokens=getattr(message.usage, 'cache_creation_input_tokens', None),
                cache_read_input_tokens=getattr(message.usage, 'cache_read_input_tokens', None)
            )

        return content, json_object, usage, None


    def create_processed_output(self) -> 'ProcessedOutput':
        """Create a ProcessedOutput from this raw output."""
        content, json_object, usage, error = self._parse_result()
        if (json_object is None and content is None) is None:
            raise ValueError("No content or JSON object found in raw output")
        if self.chat_thread_id is None:
            raise ValueError("Chat thread ID is required to create a ProcessedOutput")
        if self.chat_thread_live_id is None:
            raise ValueError("Chat thread live ID is required to create a ProcessedOutput")
    
        return ProcessedOutput(
            content=content,
            json_object=json_object,
            usage=usage,
            error=error,
            time_taken=self.time_taken,
            llm_client=self.client,
            raw_output=self,
            chat_thread_id=self.chat_thread_id,
            chat_thread_live_id=self.chat_thread_live_id
        )
    
class ProcessedOutput(Entity):
    """
    Processed and structured output from LLM interactions.
    Contains parsed content, JSON objects, and usage statistics.
    """
    content: Optional[str] = None
    json_object: Optional[GeneratedJsonObject] = None
    usage: Optional[Usage] = None
    error: Optional[str] = None
    time_taken: float
    llm_client: LLMClient
    raw_output: RawOutput
    chat_thread_id: UUID
    chat_thread_live_id: UUID
    
class ChatThread(Entity):
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
    
    forced_output: Optional[Union[StructuredTool, CallableTool]] = Field(
        default=None,
        description="Associated forced output tool"
    )
    
    llm_config: LLMConfig = Field(
        description="LLM configuration"
    )
    
    tools: List[Union[CallableTool, StructuredTool]] = Field(
        default_factory=list,
        description="Available tools"
    )
    workflow_step: Optional[int] = Field(
        default=None,
        description="Workflow step number"
    )

    @property
    def oai_response_format(self) -> Optional[Union[ResponseFormatText, ResponseFormatJSONObject, ResponseFormatJSONSchema]]:
        """Get OpenAI response format based on config."""
        if self.llm_config.response_format == ResponseFormat.text:
            return ResponseFormatText(type="text")
        elif self.llm_config.response_format == ResponseFormat.json_object:
            return ResponseFormatJSONObject(type="json_object")
        elif self.llm_config.response_format == ResponseFormat.structured_output and isinstance(self.forced_output, StructuredTool):
            assert self.forced_output is not None, "Structured output is not set"
            return self.forced_output.get_openai_json_schema_response()
        elif self.llm_config.response_format == ResponseFormat.tool and isinstance(self.forced_output, CallableTool):
            return 
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
        if self.use_schema_instruction and self.forced_output and isinstance(self.forced_output, StructuredTool):
            content = "\n".join([content, self.forced_output.schema_instruction])
        if self.llm_config.reasoner and self.llm_config.client == LLMClient.openai:
            return {"role": "developer", "content": content} if content else None
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
    def oai_messages(self) -> List[Dict[str, Any]]:
        """Convert chat history to OpenAI message format."""
        logger.info(f"Converting ChatThread({self.id}) history to OpenAI format")
        messages = []
        
        if self.system_prompt and not self.llm_config.reasoner:
            messages.append({
                "role": "system",
                "content": self.system_prompt.content
            })
        elif self.system_prompt and self.llm_config.reasoner:
            messages.append({
                "role": "developer",
                "content": self.system_prompt.content
            })
        
        for msg in self.history:
            logger.info(f"Processing message: role={msg.role}, "
                                      f"tool_call_id={msg.oai_tool_call_id}")
            
            if msg.role == MessageRole.user:
                messages.append({
                    "role": "user",
                    "content": msg.content
                })
                
            elif msg.role == MessageRole.assistant:
                if msg.tool_call:
                    messages.append({
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [{
                            "id": msg.oai_tool_call_id,
                            "type": "function",
                            "function": {
                                "name": msg.tool_name,
                                "arguments": json.dumps(msg.tool_call)
                            }
                        }]
                    })
                else:
                    messages.append({
                        "role": "assistant",
                        "content": msg.content
                    })
                
            elif msg.role == MessageRole.tool:
                messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.oai_tool_call_id
                })
        
        logger.info(f"Final messages for ChatThread({self.id}): {messages}")
        return messages

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
        if self.forced_output and self.forced_output.name == tool_name:
            return self.forced_output
        return None
    
    # @entity_uuid_expander("self")
    @entity_tracer
    def add_user_message(self) -> Optional[ChatMessage]:
        """Add a user message to history."""
        logger.debug(f"ChatThread({self.id}): Starting add_user_message")
        
        if not self.new_message and self.llm_config.response_format not in [ResponseFormat.auto_tools, ResponseFormat.workflow]:
            logger.error(f"ChatThread({self.id}): Cannot add user message - no new message content")
            raise ValueError("Cannot add user message - no new message content")
        elif not self.new_message:
            logger.info(f"ChatThread({self.id}): Skipping user message - in {self.llm_config.response_format} mode")
            return None

        parent_id = self.history[-1].id if self.history else None
        logger.debug(f"ChatThread({self.id}): Creating user message with parent_id: {parent_id}")

        user_message = ChatMessage(
            role=MessageRole.user,
            content=self.new_message,
            chat_thread_id=self.id,
            parent_message_uuid=parent_id
        )
        
        logger.info(f"ChatThread({self.id}): Created user message({user_message.id})")
        logger.debug(f"ChatThread({self.id}): Message content: {self.new_message[:100]}...")
        
        self.history.append(user_message)
        logger.info(f"ChatThread({self.id}): Added user message to history. New history length: {len(self.history)}")
        
        self.new_message = None
        logger.debug(f"ChatThread({self.id}): Cleared new_message buffer")
        
        return user_message
    
    @entity_tracer
    def reset_workflow_step(self):
        """Reset the workflow step to 0"""
        self.workflow_step = 0
        logger.info(f"ChatThread({self.id}): Reset workflow step to 0")

    @entity_tracer
    async def add_chat_turn_history(self, output: ProcessedOutput) -> Tuple[ChatMessage, ChatMessage]:
        """Add a chat turn to history, including any tool executions or validations."""
        logger.debug(f"ChatThread({self.id}): Starting add_chat_turn_history with ProcessedOutput({output.id})")
        
        # Get the parent message
        
        if not self.history:
            logger.error(f"ChatThread({self.id}): Cannot add chat turn to empty history")
            raise ValueError("Cannot add chat turn to empty history")
        
        parent_message = self.history[-1]
        logger.debug(f"ChatThread({self.id}): Parent message({parent_message.id}) role: {parent_message.role}")
        
        # Create assistant message
        logger.debug(f"ChatThread({self.id}): Creating assistant message")
        assistant_message = ChatMessage(
            role=MessageRole.assistant,
            content=output.content or "",
            chat_thread_id=self.id,
            parent_message_uuid=parent_message.id,
            tool_call=output.json_object.object if output.json_object else None,
            tool_name=output.json_object.name if output.json_object else None,
            tool_uuid=self.forced_output.id if self.forced_output else None,
            tool_type="Structured" if isinstance(self.forced_output, StructuredTool) else "Callable" if isinstance(self.forced_output, CallableTool) else None,
            oai_tool_call_id=output.json_object.tool_call_id if output.json_object else None,
            tool_json_schema=self.forced_output.json_schema if self.forced_output and isinstance(self.forced_output, StructuredTool) else None,
            usage=output.usage
        )
        
        self.history.append(assistant_message)
        logger.info(f"ChatThread({self.id}): Added assistant message({assistant_message.id})")
        logger.debug(f"ChatThread({self.id}): Assistant message details - tool_name: {assistant_message.tool_name}, tool_call_id: {assistant_message.oai_tool_call_id}")
        
        # Handle tool execution/validation
        if output.json_object and assistant_message:
            tool = self.get_tool_by_name(output.json_object.name)
            if tool:
                logger.info(f"ChatThread({self.id}): Processing tool {tool.name} ({type(tool).__name__})")
                try:
                    if isinstance(tool, StructuredTool):
                        logger.debug(f"ChatThread({self.id}): Validating structured tool input")
                        validation_result = tool.execute(input_data=output.json_object.object)
                        tool_message = ChatMessage(
                            role=MessageRole.tool,
                            content=json.dumps({"status": "validated", "message": "Schema validation successful"}),
                            chat_thread_id=self.id,
                            tool_name=tool.name,
                            tool_uuid=tool.id,
                            tool_type="Structured",
                            tool_json_schema=tool.json_schema,
                            parent_message_uuid=assistant_message.id,
                            oai_tool_call_id=assistant_message.oai_tool_call_id
                        )
                        logger.info(f"ChatThread({self.id}): Validation successful")
                    
                    elif isinstance(tool, CallableTool):
                        logger.debug(f"ChatThread({self.id}): Executing callable tool")
                        tool_result = await tool.aexecute(input_data=output.json_object.object)
                        tool_message = ChatMessage(
                            role=MessageRole.tool,
                            content=json.dumps(tool_result),
                            chat_thread_id=self.id,
                            tool_name=tool.name,
                            tool_uuid=tool.id,
                            tool_type="Callable",
                            parent_message_uuid=assistant_message.id,
                            oai_tool_call_id=assistant_message.oai_tool_call_id
                        )
                        logger.info(f"ChatThread({self.id}): Tool execution successful")
                    

                    
                    self.history.append(tool_message)
                    logger.info(f"ChatThread({self.id}): Added tool message({tool_message.id})")
                    logger.debug(f"ChatThread({self.id}): Tool message details - type: {tool_message.tool_type}, call_id: {tool_message.oai_tool_call_id}")
                    if self.workflow_step is not None and self.llm_config.response_format == ResponseFormat.workflow:
                        "not checking if the excuted tool is the tool that was supposed to be executed - no idea how could this be tesxted"
                        self.workflow_step += 1
                except Exception as e:
                    logger.error(f"ChatThread({self.id}): Tool operation failed: {str(e)}")
                    error_message = ChatMessage(
                        role=MessageRole.tool,
                        content=json.dumps({"error": str(e)}),
                        chat_thread_id=self.id,
                        tool_name=tool.name,
                        tool_uuid=tool.id,
                        tool_type="Callable" if isinstance(tool, CallableTool) else "Structured",
                        parent_message_uuid=assistant_message.id,
                        oai_tool_call_id=assistant_message.oai_tool_call_id
                    )
                    self.history.append(error_message)
                    logger.info(f"ChatThread({self.id}): Added error message({error_message.id})")
        
        logger.debug(f"ChatThread({self.id}): Completed add_chat_turn_history")
        return parent_message, assistant_message
    
    def get_tools_for_llm(self) -> Optional[List[Union[ChatCompletionToolParam, ToolParam]]]:
        """Get tools in format appropriate for current LLM."""
        if not self.tools:
            return None
            
        tools = []
        for idx, tool in enumerate(self.tools):
            # Add OpenRouter to the OpenAI-compatible clients
            if self.llm_config.client in [LLMClient.openai, LLMClient.vllm, LLMClient.litellm, LLMClient.openrouter]:
                tools.append(tool.get_openai_tool())
            elif self.llm_config.client == LLMClient.anthropic:
                if idx == 0 and self.llm_config.use_cache:
                    use_cache = True
                else:
                    use_cache = False
                tools.append(tool.get_anthropic_tool(use_cache=use_cache))
        return tools if tools else None


    @model_validator(mode='after')
    def validate_workflow(self) -> Self:
        """Validate workflow step configuration."""
        # Initialize workflow step if using workflow response format
        if self.llm_config.response_format == ResponseFormat.workflow:
            if self.workflow_step is None:
                self.workflow_step = 0
            
            # Validate step number against tools list
            if self.workflow_step >= len(self.tools):
                raise ValueError(
                    f"Workflow step {self.workflow_step} is out of range - only {len(self.tools)} tools available"
                )
        
        return self
    
    def get_all_usages(self) -> List[Usage]:
        """Get all usages from history"""
        usages = []
        for message in self.history:
            if message.role == MessageRole.assistant and message.usage:
                usages.append(message.usage)
        return usages

    @model_validator(mode='before')
    def validate_history(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-validate to ensure history contains proper ChatMessage objects"""
        if 'history' in values and isinstance(values['history'], list):
            history = []
            for msg in values['history']:
                if isinstance(msg, dict):
                    msg = ChatMessage.model_validate(msg)
                history.append(msg)
            values['history'] = history
        return values
    
    def _apply_modifications_and_create_version(self, cold_snapshot: 'Entity', force: bool, **kwargs) -> bool:
        """ calls the Entity class apply modifications and adds a patch to the message chat thread history ids"""
        super()._apply_modifications_and_create_version(cold_snapshot, force, **kwargs)
        new_parent_message_uuid = None
        for message in self.history:
            message.fork(chat_thread_id = self.id, parent_message_uuid = new_parent_message_uuid if new_parent_message_uuid else message.parent_message_uuid)
            new_parent_message_uuid = message.id
        return True





