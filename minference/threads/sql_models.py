"""
Refactored SQL models with:
- Auto-assigned table names (removing explicit table=True where necessary)
- Integer primary keys for database entries
- UUID-based `ecs_id` fields for semantic versioning
- Eagerly loaded relationships with lazy="joined"
- Bidirectional many-to-many relationships
- to_entity()/from_entity() conversion methods
- Relationship handler methods that work with the storage class
"""

from typing import List, Optional, Dict, Any, Union, Literal, Type, cast, Set
from uuid import UUID, uuid4
from datetime import datetime, UTC
import json
import logging

from sqlmodel import SQLModel, Field, Relationship, Session, create_engine, select
from sqlalchemy import Column, JSON
from sqlalchemy.types import String as SQLAlchemyString
from enum import Enum

from minference.ecs.entity import Entity, SQLModelType
from minference.threads.models import (
    ChatThread, ChatMessage, LLMConfig, Usage, GeneratedJsonObject,
    CallableTool, StructuredTool, RawOutput, ProcessedOutput, SystemPrompt,
    MessageRole, LLMClient, ToolType, ResponseFormat
)

###############################################################################
# Utility Functions
###############################################################################

def convert_uuids_to_strings(data: Any) -> Any:
    """
    Recursively convert all UUIDs to strings in any data structure.
    Works with dicts, lists, and nested combinations.
    """
    if isinstance(data, UUID):
        return str(data)
    elif isinstance(data, dict):
        return {k: convert_uuids_to_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_uuids_to_strings(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_uuids_to_strings(item) for item in data)
    elif isinstance(data, set):
        return {convert_uuids_to_strings(item) for item in data}
    else:
        return data

def modify_entity_for_sql(entity: Entity) -> Dict[str, Any]:
    """
    Modify entity fields for SQL storage, converting UUIDs to strings in JSON fields
    and handling other common conversions.
    
    Returns a dictionary of the modified fields.
    """
    # Start with the base fields that are common to all entities
    result = {
        "ecs_id": entity.ecs_id,
        "lineage_id": entity.lineage_id,
        "parent_id": entity.parent_id,
        "created_at": entity.created_at,
        "old_ids": convert_uuids_to_strings(entity.old_ids or [])
    }
    
    return result

###############################################################################
# Type Definitions 
###############################################################################

# Define Literal types based on the enums
MessageRoleLiteral = Literal["user", "assistant", "tool", "system", "developer"]
LLMClientLiteral = Literal["openai", "azure_openai", "anthropic", "vllm", "litellm", "openrouter"]
ResponseFormatLiteral = Literal["json_beg", "text", "json_object", "structured_output", "tool", "auto_tools", "workflow"]
ToolTypeLiteral = Literal["Callable", "Structured"]

###############################################################################
# Bridging Tables (Many-to-Many)
###############################################################################

class ThreadMessageLink(SQLModel, table=True):
    """Many-to-many link between ChatThreadSQL and ChatMessageSQL."""
    thread_id: int = Field(foreign_key="chatthreadsql.id", primary_key=True)
    message_id: int = Field(foreign_key="chatmessagesql.id", primary_key=True)

class ThreadToolLink(SQLModel, table=True):
    """Many-to-many link between ChatThreadSQL and ToolSQL."""
    thread_id: int = Field(foreign_key="chatthreadsql.id", primary_key=True)
    tool_id: int = Field(foreign_key="toolsql.id", primary_key=True)

class OutputJsonObjectLink(SQLModel, table=True):
    """Link model between ProcessedOutput and GeneratedJsonObject"""
    processed_output_id: int = Field(foreign_key="processedoutputsql.id", primary_key=True)
    json_object_id: int = Field(foreign_key="generatedjsonobjectsql.id", primary_key=True)

###############################################################################
# UsageSQL
###############################################################################

class UsageSQL(SQLModel, table=True):
    """SQL model for Usage"""
    # Database primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Entity versioning
    ecs_id: UUID = Field(default_factory=uuid4, index=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    old_ids: List[UUID] = Field(default_factory=list, sa_column=Column(JSON))

    # Usage metrics
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    model: str
    accepted_prediction_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None

    # Relationships
    message: Optional["ChatMessageSQL"] = Relationship(
        back_populates="usage",
        sa_relationship_kwargs={"uselist": False, "lazy": "joined"}
    )
    
    processed_output: Optional["ProcessedOutputSQL"] = Relationship(
        back_populates="usage",
        sa_relationship_kwargs={"uselist": False, "lazy": "joined"}
    )

    def to_entity(self) -> Usage:
        """Convert SQL model to domain entity"""
        return Usage(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=self.old_ids,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens,
            model=self.model,
            accepted_prediction_tokens=self.accepted_prediction_tokens,
            audio_tokens=self.audio_tokens,
            reasoning_tokens=self.reasoning_tokens,
            rejected_prediction_tokens=self.rejected_prediction_tokens,
            cached_tokens=self.cached_tokens,
            from_storage=True
        )

    @classmethod
    def from_entity(cls, entity: Usage) -> "UsageSQL":
        """Create SQL model from domain entity"""
        base_fields = modify_entity_for_sql(entity)
        return cls(
            **base_fields,
            prompt_tokens=entity.prompt_tokens,
            completion_tokens=entity.completion_tokens,
            total_tokens=entity.total_tokens,
            cache_creation_input_tokens=entity.cache_creation_input_tokens,
            cache_read_input_tokens=entity.cache_read_input_tokens,
            model=entity.model,
            accepted_prediction_tokens=entity.accepted_prediction_tokens,
            audio_tokens=entity.audio_tokens,
            reasoning_tokens=entity.reasoning_tokens,
            rejected_prediction_tokens=entity.rejected_prediction_tokens,
            cached_tokens=entity.cached_tokens
        )
    
    def handle_relationships(self, entity: Usage, session: Session, orm_objects: Optional[Dict[UUID, Any]] = None) -> None:
        """Generic relationship handler"""
        # Use an empty dict if orm_objects is None
        orm_objects = orm_objects or {}
        
        # Usage doesn't have complex relationships to handle
        pass

###############################################################################
# GeneratedJsonObjectSQL
###############################################################################

class GeneratedJsonObjectSQL(SQLModel, table=True):
    """SQL model for GeneratedJsonObject"""
    # Database primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Entity versioning
    ecs_id: UUID = Field(default_factory=uuid4, index=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    old_ids: List[UUID] = Field(default_factory=list, sa_column=Column(JSON))
    
    # Domain fields
    name: str
    object: Dict[str, Any] = Field(sa_column=Column(JSON))
    tool_call_id: Optional[str] = None

    # Relationships
    processed_output: Optional["ProcessedOutputSQL"] = Relationship(
        back_populates="json_object",
        link_model=OutputJsonObjectLink,
        sa_relationship_kwargs={"lazy": "joined"}
    )

    def to_entity(self) -> GeneratedJsonObject:
        """Convert to domain entity"""
        return GeneratedJsonObject(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=self.old_ids,
            name=self.name,
            object=self.object,
            tool_call_id=self.tool_call_id,
            from_storage=True
        )

    @classmethod
    def from_entity(cls, entity: GeneratedJsonObject) -> "GeneratedJsonObjectSQL":
        """Create from domain entity"""
        base_fields = modify_entity_for_sql(entity)
        return cls(
            **base_fields,
            name=entity.name,
            object=convert_uuids_to_strings(entity.object),
            tool_call_id=entity.tool_call_id
        )
    
    def handle_relationships(self, entity: GeneratedJsonObject, session: Session, orm_objects: Optional[Dict[UUID, Any]] = None) -> None:
        """Generic relationship handler"""
        # Use an empty dict if orm_objects is None
        orm_objects = orm_objects or {}
        
        # GeneratedJsonObject doesn't have complex relationships to handle
        pass

###############################################################################
# SystemPromptSQL
###############################################################################

class SystemPromptSQL(SQLModel, table=True):
    """SQL model for the SystemPrompt entity."""
    # Database primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Entity versioning
    ecs_id: UUID = Field(default_factory=uuid4, index=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    old_ids: List[UUID] = Field(default_factory=list, sa_column=Column(JSON))

    # Domain fields
    name: str
    content: str

    # Relationships
    threads: List["ChatThreadSQL"] = Relationship(
        back_populates="system_prompt",
        sa_relationship_kwargs={"lazy": "joined"}
    )

    def to_entity(self) -> SystemPrompt:
        """Convert to domain entity"""
        # The domain model expects these parameters to be in a specific order or different format
        # Explicitly create a dictionary of parameters to pass to the constructor
        params = {
            "ecs_id": self.ecs_id,
            "lineage_id": self.lineage_id,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "old_ids": self.old_ids,
            "name": self.name,
            "content": self.content,
            "from_storage": True
        }
        return SystemPrompt(**params)

    @classmethod
    def from_entity(cls, entity: SystemPrompt) -> "SystemPromptSQL":
        """Create from domain entity"""
        base_fields = modify_entity_for_sql(entity)
        return cls(
            **base_fields,
            name=entity.name,
            content=entity.content
        )
    
    def handle_relationships(self, entity: SystemPrompt, session: Session, orm_objects: Optional[Dict[UUID, Any]] = None) -> None:
        """Generic relationship handler"""
        # Use an empty dict if orm_objects is None
        orm_objects = orm_objects or {}
        
        # SystemPrompt doesn't have complex relationships to handle
        pass

###############################################################################
# LLMConfigSQL
###############################################################################

class LLMConfigSQL(SQLModel, table=True):
    """SQL model for the LLMConfig entity."""
    # Database primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Entity versioning
    ecs_id: UUID = Field(default_factory=uuid4, index=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    old_ids: List[UUID] = Field(default_factory=list, sa_column=Column(JSON))

    # Store enums as strings
    client: str = Field(sa_column=Column(SQLAlchemyString))
    model: Optional[str] = None
    max_tokens: int = 400
    temperature: float = 0
    response_format: str = Field(sa_column=Column(SQLAlchemyString))
    use_cache: bool = True
    reasoner: bool = False
    reasoning_effort: str = Field(default="medium", sa_column=Column(SQLAlchemyString))

    # Relationships
    threads: List["ChatThreadSQL"] = Relationship(
        back_populates="llm_config",
        sa_relationship_kwargs={"lazy": "joined"}
    )

    def to_entity(self) -> LLMConfig:
        """Convert to domain entity"""
        return LLMConfig(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=self.old_ids,
            client=LLMClient(self.client),  # Convert string to enum
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format=ResponseFormat(self.response_format),  # Convert string to enum
            use_cache=self.use_cache,
            reasoner=self.reasoner,
            reasoning_effort=cast(Literal["low", "medium", "high"], self.reasoning_effort),  # Cast to proper type
            from_storage=True
        )

    @classmethod
    def from_entity(cls, entity: LLMConfig) -> "LLMConfigSQL":
        """Create from domain entity"""
        base_fields = modify_entity_for_sql(entity)
        return cls(
            **base_fields,
            client=entity.client.value,  # Store enum value as string
            model=entity.model,
            max_tokens=entity.max_tokens,
            temperature=entity.temperature,
            response_format=entity.response_format.value,  # Store enum value as string
            use_cache=entity.use_cache,
            reasoner=entity.reasoner,
            reasoning_effort=entity.reasoning_effort
        )
    
    def handle_relationships(self, entity: LLMConfig, session: Session, orm_objects: Optional[Dict[UUID, Any]] = None) -> None:
        """Generic relationship handler"""
        # Use an empty dict if orm_objects is None
        orm_objects = orm_objects or {}
        
        # LLMConfig doesn't have complex relationships to handle
        pass

###############################################################################
# ToolSQL (Unifying CallableTool & StructuredTool)
###############################################################################

class ToolSQL(SQLModel, table=True):
    """SQL model for Tool entity"""
    # Database primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Entity versioning
    ecs_id: UUID = Field(default_factory=uuid4, index=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    old_ids: List[UUID] = Field(default_factory=list, sa_column=Column(JSON))

    # Basic fields
    name: str
    
    # Tool type specific fields
    json_schema: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))  # For StructuredTool
    description: Optional[str] = None  # For StructuredTool
    instruction_string: Optional[str] = None  # For StructuredTool
    docstring: Optional[str] = None  # For CallableTool
    callable_text: Optional[str] = None  # For CallableTool
    input_schema: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))  # For CallableTool
    output_schema: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))  # For CallableTool
    strict_schema: bool = True  # For both types

    # Relationships
    threads: List["ChatThreadSQL"] = Relationship(
        back_populates="tools",
        link_model=ThreadToolLink,
        sa_relationship_kwargs={"lazy": "joined"}
    )
    
    as_forced_output: List["ChatThreadSQL"] = Relationship(
        back_populates="forced_output",
        sa_relationship_kwargs={"lazy": "joined"}
    )

    def to_entity(self) -> Union[CallableTool, StructuredTool]:
        """Convert to domain entity"""
        base_args = {
            "ecs_id": self.ecs_id,
            "lineage_id": self.lineage_id,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "old_ids": self.old_ids,
            "name": self.name,
            "strict_schema": self.strict_schema,
            "from_storage": True
        }
        
        if self.callable_text is not None:
            return CallableTool(
                **base_args,
                docstring=self.docstring,
                input_schema=self.input_schema,
                output_schema=self.output_schema,
                callable_text=self.callable_text
            )
        else:
            return StructuredTool(
                **base_args,
                description=self.description or "",
                instruction_string=self.instruction_string or "Please follow this JSON schema for your response:",
                json_schema=self.json_schema or {}
            )

    @classmethod
    def from_entity(cls, entity: Union[CallableTool, StructuredTool]) -> "ToolSQL":
        """Create from domain entity"""
        base_fields = modify_entity_for_sql(entity)
        
        # Common fields for both tool types
        common_fields = {
            **base_fields,
            "name": entity.name,
            "strict_schema": entity.strict_schema,
        }
        
        if isinstance(entity, CallableTool):
            return cls(
                **common_fields,
                docstring=entity.docstring,
                input_schema=convert_uuids_to_strings(entity.input_schema),
                output_schema=convert_uuids_to_strings(entity.output_schema),
                callable_text=entity.callable_text,
                json_schema=None,
                description=None,
                instruction_string=None
            )
        else:
            return cls(
                **common_fields,
                description=entity.description,
                instruction_string=entity.instruction_string,
                json_schema=convert_uuids_to_strings(entity.json_schema),
                callable_text=None,
                docstring=None,
                input_schema={},
                output_schema={}
            )
    
    def handle_relationships(self, entity: Union[CallableTool, StructuredTool], session: Session, orm_objects: Optional[Dict[UUID, Any]] = None) -> None:
        """Generic relationship handler"""
        # Use an empty dict if orm_objects is None
        orm_objects = orm_objects or {}
        
        # Tool doesn't have complex relationships to handle beyond what's in link tables
        pass

###############################################################################
# ChatMessageSQL
###############################################################################

class ChatMessageSQL(SQLModel, table=True):
    """SQL model for a single ChatMessage entity version."""
    # Database primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Entity versioning
    ecs_id: UUID = Field(default_factory=uuid4, index=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    old_ids: List[UUID] = Field(default_factory=list, sa_column=Column(JSON))

    # Message threading field (different from versioning)
    parent_message_uuid: Optional[UUID] = Field(default=None)

    # Message content
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    role: str = Field(sa_column=Column(SQLAlchemyString))
    content: str
    author_uuid: Optional[UUID] = None
    chat_thread_id: Optional[UUID] = None

    # Tool-related fields
    tool_name: Optional[str] = None
    tool_uuid: Optional[UUID] = None
    tool_type: Optional[ToolTypeLiteral] = Field(default=None, sa_column=Column(SQLAlchemyString))
    oai_tool_call_id: Optional[str] = None
    tool_json_schema: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    tool_call: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # Bidirectional relationship with UsageSQL
    usage_id: Optional[int] = Field(default=None, foreign_key="usagesql.id")
    usage: Optional[UsageSQL] = Relationship(
        back_populates="message",
        sa_relationship_kwargs={"uselist": False, "lazy": "joined"}
    )

    # Relationship to threads (M-to-M)
    threads: List["ChatThreadSQL"] = Relationship(
        back_populates="messages",
        link_model=ThreadMessageLink,
        sa_relationship_kwargs={"lazy": "joined"}
    )

    def to_entity(self) -> ChatMessage:
        """Convert SQL model to domain entity"""
        return ChatMessage(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=self.old_ids,
            parent_message_uuid=self.parent_message_uuid,  # Threading parent
            timestamp=self.timestamp,
            role=MessageRole(self.role),
            content=self.content,
            author_uuid=self.author_uuid,
            chat_thread_id=self.chat_thread_id,
            tool_name=self.tool_name,
            tool_uuid=self.tool_uuid,
            tool_type=self.tool_type,
            oai_tool_call_id=self.oai_tool_call_id,
            tool_json_schema=self.tool_json_schema,
            tool_call=self.tool_call,
            usage=self.usage.to_entity() if self.usage else None,
            from_storage=True
        )

    @classmethod
    def from_entity(cls, entity: ChatMessage) -> "ChatMessageSQL":
        """Create SQL model from domain entity"""
        base_fields = modify_entity_for_sql(entity)
        message = cls(
            **base_fields,
            parent_message_uuid=entity.parent_message_uuid,
            timestamp=entity.timestamp,
            role=entity.role.value,
            content=entity.content,
            author_uuid=entity.author_uuid,
            chat_thread_id=entity.chat_thread_id,
            tool_name=entity.tool_name,
            tool_uuid=entity.tool_uuid,
            tool_type=entity.tool_type,
            oai_tool_call_id=entity.oai_tool_call_id,
            tool_json_schema=convert_uuids_to_strings(entity.tool_json_schema),
            tool_call=convert_uuids_to_strings(entity.tool_call)
        )
        
        # We don't set usage here - it will be handled in the relationship handler
        return message
    
    def handle_relationships(self, entity: ChatMessage, session: Session, orm_objects: Optional[Dict[UUID, Any]] = None) -> None:
        """Generic relationship handler"""
        # Use an empty dict if orm_objects is None
        orm_objects = orm_objects or {}
        
        # Handle usage relationship
        if entity.usage and entity.usage.ecs_id in orm_objects:
            self.usage = orm_objects[entity.usage.ecs_id]

###############################################################################
# RawOutputSQL
###############################################################################

class RawOutputSQL(SQLModel, table=True):
    """SQL model for RawOutput"""
    # Database primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Entity versioning
    ecs_id: UUID = Field(default_factory=uuid4, index=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    old_ids: List[UUID] = Field(default_factory=list, sa_column=Column(JSON))
    
    # Domain fields
    raw_result: Dict[str, Any] = Field(sa_column=Column(JSON))
    completion_kwargs: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    chat_thread_id: Optional[UUID] = Field(default=None)
    chat_thread_live_id: Optional[UUID] = None
    start_time: float
    end_time: float
    client: str = Field(sa_column=Column(SQLAlchemyString))

    # Relationships
    processed_outputs: List["ProcessedOutputSQL"] = Relationship(
        back_populates="raw_output",
        sa_relationship_kwargs={"lazy": "joined"}
    )

    def to_entity(self) -> RawOutput:
        """Convert to domain entity"""
        return RawOutput(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=self.old_ids,
            raw_result=self.raw_result,
            completion_kwargs=self.completion_kwargs,
            chat_thread_id=self.chat_thread_id,
            chat_thread_live_id=self.chat_thread_live_id,
            start_time=self.start_time,
            end_time=self.end_time,
            client=LLMClient(self.client),
            from_storage=True
        )

    @classmethod
    def from_entity(cls, entity: RawOutput) -> "RawOutputSQL":
        """Create from domain entity"""
        base_fields = modify_entity_for_sql(entity)
        return cls(
            **base_fields,
            raw_result=convert_uuids_to_strings(entity.raw_result),
            completion_kwargs=convert_uuids_to_strings(entity.completion_kwargs or {}),
            chat_thread_id=entity.chat_thread_id,
            chat_thread_live_id=entity.chat_thread_live_id,
            start_time=entity.start_time,
            end_time=entity.end_time,
            client=entity.client.value
        )
    
    def handle_relationships(self, entity: RawOutput, session: Session, orm_objects: Optional[Dict[UUID, Any]] = None) -> None:
        """Generic relationship handler"""
        # Use an empty dict if orm_objects is None
        orm_objects = orm_objects or {}
        
        # RawOutput doesn't have complex relationships to handle
        pass

###############################################################################
# ProcessedOutputSQL
###############################################################################

# In the ProcessedOutputSQL class, we need to fix the chat_thread relationship
# Replace the current definition with this:

class ProcessedOutputSQL(SQLModel, table=True):
    """SQL model for the ProcessedOutput entity."""
    # Database primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Entity versioning
    ecs_id: UUID = Field(default_factory=uuid4, index=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    old_ids: List[UUID] = Field(default_factory=list, sa_column=Column(JSON))

    # Domain fields
    content: Optional[str] = None
    error: Optional[str] = None
    time_taken: float
    llm_client: str = Field(sa_column=Column(SQLAlchemyString))
    # UUID version of chat_thread_id (from entity)
    chat_thread_uuid: UUID
    chat_thread_live_id: UUID

    # Integer foreign key to ChatThreadSQL
    chat_thread_id: Optional[int] = Field(default=None, foreign_key="chatthreadsql.id")
    
    # Relationships
    json_object_id: Optional[int] = Field(default=None, foreign_key="generatedjsonobjectsql.id")
    json_object: Optional[GeneratedJsonObjectSQL] = Relationship(
        back_populates="processed_output",
        link_model=OutputJsonObjectLink,
        sa_relationship_kwargs={"lazy": "joined"}
    )
    
    usage_id: Optional[int] = Field(default=None, foreign_key="usagesql.id")
    usage: Optional[UsageSQL] = Relationship(
        back_populates="processed_output",
        sa_relationship_kwargs={"lazy": "joined"}
    )
    
    raw_output_id: Optional[int] = Field(default=None, foreign_key="rawoutputsql.id")
    raw_output: Optional[RawOutputSQL] = Relationship(
        back_populates="processed_outputs",
        sa_relationship_kwargs={"lazy": "joined"}
    )
    
    # Relationship to chat thread
    chat_thread: Optional["ChatThreadSQL"] = Relationship(
        back_populates="processed_outputs",
        sa_relationship_kwargs={"lazy": "joined"}
    )

    def to_entity(self) -> ProcessedOutput:
        """Convert to domain entity"""
        if not self.raw_output:
            raise ValueError("ProcessedOutput requires a RawOutput to be present in DB.")

        return ProcessedOutput(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=self.old_ids,
            content=self.content,
            json_object=self.json_object.to_entity() if self.json_object else None,
            usage=self.usage.to_entity() if self.usage else None,
            error=self.error,
            time_taken=self.time_taken,
            llm_client=LLMClient(self.llm_client),
            raw_output=self.raw_output.to_entity(),
            chat_thread_id=self.chat_thread_uuid,  # Use UUID version
            chat_thread_live_id=self.chat_thread_live_id,
            from_storage=True
        )

    @classmethod
    def from_entity(cls, entity: ProcessedOutput) -> "ProcessedOutputSQL":
        """Create from domain entity"""
        # Create the basic instance without relationships
        base_fields = modify_entity_for_sql(entity)
        output = cls(
            **base_fields,
            content=entity.content,
            error=entity.error,
            time_taken=entity.time_taken,
            llm_client=entity.llm_client.value,
            chat_thread_uuid=entity.chat_thread_id,  # Store UUID version
            chat_thread_live_id=entity.chat_thread_live_id
        )
        
        # We don't set relationships here - they will be handled in handle_relationships
        return output
    
    def handle_relationships(self, entity: ProcessedOutput, session: Session, orm_objects: Optional[Dict[UUID, Any]] = None) -> None:
        """Generic relationship handler"""
        # Use an empty dict if orm_objects is None
        orm_objects = orm_objects or {}
        
        # Handle json_object relationship
        if entity.json_object and entity.json_object.ecs_id in orm_objects:
            self.json_object = orm_objects[entity.json_object.ecs_id]
            
        # Handle usage relationship
        if entity.usage and entity.usage.ecs_id in orm_objects:
            self.usage = orm_objects[entity.usage.ecs_id]
            
        # Handle raw_output relationship
        if entity.raw_output and entity.raw_output.ecs_id in orm_objects:
            self.raw_output = orm_objects[entity.raw_output.ecs_id]
            
        # Try to find the ChatThreadSQL instance by ecs_id
        if entity.chat_thread_id:
            # Look for ChatThreadSQL in orm_objects
            for obj_id, orm_obj in orm_objects.items():
                if isinstance(orm_obj, ChatThreadSQL) and orm_obj.ecs_id == entity.chat_thread_id:
                    self.chat_thread = orm_obj
                    self.chat_thread_id = orm_obj.id  # Set the integer foreign key
                    break

###############################################################################
# ChatThreadSQL
###############################################################################

class ChatThreadSQL(SQLModel, table=True):
    """SQL model for the ChatThread entity."""
    # Database primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Entity versioning
    ecs_id: UUID = Field(default_factory=uuid4, index=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    old_ids: List[UUID] = Field(default_factory=list, sa_column=Column(JSON))

    # Domain fields
    name: Optional[str] = None
    new_message: Optional[str] = None
    prefill: str = "Here's the valid JSON object response:```json"
    postfill: str = "\n\nPlease provide your response in JSON format."
    use_schema_instruction: bool = False
    use_history: bool = True
    workflow_step: Optional[int] = None

    # Relationship to system prompt
    system_prompt_id: Optional[int] = Field(default=None, foreign_key="systempromptsql.id")
    system_prompt: Optional[SystemPromptSQL] = Relationship(
        back_populates="threads",
        sa_relationship_kwargs={"lazy": "joined"}
    )

    # Relationship to LLMConfig
    llm_config_id: Optional[int] = Field(default=None, foreign_key="llmconfigsql.id")
    llm_config: Optional[LLMConfigSQL] = Relationship(
        back_populates="threads", 
        sa_relationship_kwargs={"lazy": "joined"}
    )

    # Relationship to messages (M-to-M)
    messages: List[ChatMessageSQL] = Relationship(
        back_populates="threads",
        link_model=ThreadMessageLink,
        sa_relationship_kwargs={"lazy": "joined", "order_by": "ChatMessageSQL.timestamp"}
    )

    # Relationship to tools (M-to-M)
    tools: List[ToolSQL] = Relationship(
        back_populates="threads",
        link_model=ThreadToolLink,
        sa_relationship_kwargs={"lazy": "joined"}
    )

    # Relationship to forced_output tool
    forced_output_id: Optional[int] = Field(default=None, foreign_key="toolsql.id")
    forced_output: Optional[ToolSQL] = Relationship(
        back_populates="as_forced_output",
        sa_relationship_kwargs={"lazy": "joined"}
    )
    
    # ProcessedOutputs
    processed_outputs: List[ProcessedOutputSQL] = Relationship(
        sa_relationship_kwargs={"lazy": "joined"}
    )

    def to_entity(self) -> ChatThread:
        """Convert to domain entity"""
        # Validate required relationships
        if not self.llm_config:
            raise ValueError("ChatThread requires an LLMConfig to be present in DB.")
            
        # Convert related objects
        message_entities = [m.to_entity() for m in self.messages]
        system_prompt_entity = self.system_prompt.to_entity() if self.system_prompt else None
        llm_config_entity = self.llm_config.to_entity()
        tool_entities = [t.to_entity() for t in self.tools]
        forced_output_entity = self.forced_output.to_entity() if self.forced_output else None

        return ChatThread(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=self.old_ids,
            name=self.name,
            system_prompt=system_prompt_entity,
            history=message_entities,
            new_message=self.new_message,
            prefill=self.prefill,
            postfill=self.postfill,
            use_schema_instruction=self.use_schema_instruction,
            use_history=self.use_history,
            forced_output=forced_output_entity,
            llm_config=llm_config_entity,
            tools=tool_entities,
            workflow_step=self.workflow_step,
            from_storage=True
        )

    @classmethod
    def from_entity(cls, entity: ChatThread) -> "ChatThreadSQL":
        """Create from domain entity"""
        # Create the basic thread instance without relationships
        base_fields = modify_entity_for_sql(entity)
        chat_thread = cls(
            **base_fields,
            name=entity.name,
            new_message=entity.new_message,
            prefill=entity.prefill,
            postfill=entity.postfill,
            use_schema_instruction=entity.use_schema_instruction,
            use_history=entity.use_history,
            workflow_step=entity.workflow_step
        )
        
        # We don't set relationships here - they will be handled in handle_relationships
        return chat_thread
    
    def handle_relationships(self, entity: ChatThread, session: Session, orm_objects: Optional[Dict[UUID, Any]] = None) -> None:
        """Generic relationship handler"""
        # Handle system_prompt relationship
        orm_objects = orm_objects or {}
        
        if entity.system_prompt and entity.system_prompt.ecs_id in orm_objects:
            self.system_prompt = orm_objects[entity.system_prompt.ecs_id]
            
        # Handle llm_config relationship
        if entity.llm_config and entity.llm_config.ecs_id in orm_objects:
            self.llm_config = orm_objects[entity.llm_config.ecs_id]
            
        # Handle forced_output relationship
        if entity.forced_output and entity.forced_output.ecs_id in orm_objects:
            self.forced_output = orm_objects[entity.forced_output.ecs_id]
            
        # Handle messages relationship
        self.add_messages_from_entity(entity, session, orm_objects)
        
        # Handle tools relationship
        self.add_tools_from_entity(entity, session, orm_objects)
    
    def add_messages_from_entity(self, entity: ChatThread, session: Session, orm_objects: Optional[Dict[UUID, Any]] = None) -> None:
        """Special handler for message relationships"""
        logger = logging.getLogger("ChatThreadSQL")
        logger.debug(f"Adding {len(entity.history)} messages to ChatThread({self.ecs_id})")
        
        # Use an empty dict if orm_objects is None
        orm_objects = orm_objects or {}
        
        self.messages.clear()  # Clear existing messages
        
        for message in entity.history:
            if message.ecs_id in orm_objects:
                message_sql = orm_objects[message.ecs_id]
                self.messages.append(message_sql)
            else:
                logger.warning(f"Message {message.ecs_id} not found in orm_objects")
    
    def add_tools_from_entity(self, entity: ChatThread, session: Session, orm_objects: Optional[Dict[UUID, Any]] = None) -> None:
        """Special handler for tool relationships"""
        logger = logging.getLogger("ChatThreadSQL")
        logger.debug(f"Adding {len(entity.tools)} tools to ChatThread({self.ecs_id})")
        
        # Use an empty dict if orm_objects is None
        orm_objects = orm_objects or {}
        
        self.tools.clear()  # Clear existing tools
        
        for tool in entity.tools:
            if tool.ecs_id in orm_objects:
                tool_sql = orm_objects[tool.ecs_id]
                self.tools.append(tool_sql)
            else:
                logger.warning(f"Tool {tool.ecs_id} not found in orm_objects")

###############################################################################
# Entity to ORM Mapping
###############################################################################

# This dictionary maps domain entity classes to their SQL ORM counterparts
# It can be imported in scripts to avoid redundant mapping definitions
ENTITY_ORM_MAP = {
    # Core chat entities
    cast(Type[Entity], ChatThread): cast(Type[SQLModelType], ChatThreadSQL),
    cast(Type[Entity], ChatMessage): cast(Type[SQLModelType], ChatMessageSQL),
    cast(Type[Entity], LLMConfig): cast(Type[SQLModelType], LLMConfigSQL),
    cast(Type[Entity], Usage): cast(Type[SQLModelType], UsageSQL),
    cast(Type[Entity], SystemPrompt): cast(Type[SQLModelType], SystemPromptSQL),
    cast(Type[Entity], GeneratedJsonObject): cast(Type[SQLModelType], GeneratedJsonObjectSQL),
    
    # Tool entities (share a table)
    cast(Type[Entity], CallableTool): cast(Type[SQLModelType], ToolSQL),
    cast(Type[Entity], StructuredTool): cast(Type[SQLModelType], ToolSQL),
    
    # Output entities
    cast(Type[Entity], RawOutput): cast(Type[SQLModelType], RawOutputSQL),
    cast(Type[Entity], ProcessedOutput): cast(Type[SQLModelType], ProcessedOutputSQL)
}