"""
Refactored SQL models with:
- UUID-based primary keys for versioning
- Bridging tables for many-to-many relationships
- to_entity()/from_entity() conversion methods
- No business logic methods - pure storage models
"""

from typing import List, Optional, Dict, Any, Union, Literal, cast
from uuid import UUID, uuid4
from datetime import datetime, UTC
import json

from sqlmodel import SQLModel, Field, Relationship, Session
from sqlalchemy import Column, JSON
from sqlalchemy.types import String as SQLAlchemyString
from enum import Enum

from minference.threads.models import (
    ChatThread, ChatMessage, LLMConfig, Usage, GeneratedJsonObject,
    CallableTool, StructuredTool, RawOutput, ProcessedOutput, SystemPrompt,
    MessageRole, LLMClient, ToolType, ResponseFormat
)

###############################################################################
# Type Definitions (No Conversion Helpers)
###############################################################################

# Define Literal types based on the enums
MessageRoleLiteral = Literal["user", "assistant", "tool", "system", "developer"]
LLMClientLiteral = Literal["openai", "azure_openai", "anthropic", "vllm", "litellm", "openrouter"]
ResponseFormatLiteral = Literal["json_beg", "text", "json_object", "structured_output", "tool", "auto_tools", "workflow"]
ToolTypeLiteral = Literal["Callable", "Structured"]

###############################################################################
# Bridging Tables (Many-to-Many)
###############################################################################

class ThreadMessageLinkSQL(SQLModel, table=True):
    """Many-to-many link between ChatThreadSQL and ChatMessageSQL."""
    thread_id: UUID = Field(foreign_key="chatthreadsql.id", primary_key=True)
    message_id: UUID = Field(foreign_key="chatmessagesql.id", primary_key=True)

class ThreadToolLinkSQL(SQLModel, table=True):
    """Many-to-many link between ChatThreadSQL and ToolSQL."""
    thread_id: UUID = Field(foreign_key="chatthreadsql.id", primary_key=True)
    tool_id: UUID = Field(foreign_key="toolsql.id", primary_key=True)

class OutputJsonObjectLinkageSQL(SQLModel, table=True):
    """Link model between ProcessedOutput and GeneratedJsonObject"""
    processed_output_id: UUID = Field(foreign_key="processedoutputsql.id", primary_key=True)
    json_object_id: UUID = Field(foreign_key="generatedjsonobjectsql.id", primary_key=True)

###############################################################################
# UsageSQL
###############################################################################

class UsageSQL(SQLModel, table=True):
    """SQL model for Usage"""
    # Primary key and versioning
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="usagesql.id")

    # Usage metrics
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    model: str

    # One-to-one relationship with ChatMessageSQL
    message: Optional["ChatMessageSQL"] = Relationship(
        back_populates="usage",
        sa_relationship_kwargs={"uselist": False}
    )

    def to_entity(self) -> Usage:
        """Convert SQL model to domain entity"""
        return Usage(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens,
            model=self.model,
            from_storage=True
        )

    @classmethod
    def from_entity(cls, entity: Usage) -> "UsageSQL":
        """Create SQL model from domain entity"""
        return cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            prompt_tokens=entity.prompt_tokens,
            completion_tokens=entity.completion_tokens,
            total_tokens=entity.total_tokens,
            cache_creation_input_tokens=entity.cache_creation_input_tokens,
            cache_read_input_tokens=entity.cache_read_input_tokens,
            model=entity.model
        )

###############################################################################
# GeneratedJsonObjectSQL
###############################################################################

class GeneratedJsonObjectSQL(SQLModel, table=True):
    """SQL model for GeneratedJsonObject"""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="generatedjsonobjectsql.id")
    name: str
    object: Dict[str, Any] = Field(sa_column=Column(JSON))
    tool_call_id: Optional[str] = None

    def to_entity(self) -> GeneratedJsonObject:
        return GeneratedJsonObject(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            name=self.name,
            object=self.object,
            tool_call_id=self.tool_call_id,
            from_storage=True
        )

    @classmethod
    def from_entity(cls, entity: GeneratedJsonObject) -> "GeneratedJsonObjectSQL":
        return cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            name=entity.name,
            object=entity.object,
            tool_call_id=entity.tool_call_id
        )

###############################################################################
# ChatMessageSQL
###############################################################################

class ChatMessageSQL(SQLModel, table=True):
    """SQL model for a single ChatMessage entity version."""
    # Primary key and versioning
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="chatmessagesql.id")

    # Message threading field (different from versioning)
    parent_message_uuid: Optional[UUID] = Field(default=None, foreign_key="chatmessagesql.id")

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
    usage_id: Optional[UUID] = Field(default=None, foreign_key="usagesql.id")
    usage: Optional[UsageSQL] = Relationship(
        back_populates="message",
        sa_relationship_kwargs={"uselist": False}
    )

    # Relationship to threads (M-to-M)
    threads: List["ChatThreadSQL"] = Relationship(
        back_populates="messages",
        link_model=ThreadMessageLinkSQL
    )

    def to_entity(self) -> ChatMessage:
        """Convert SQL model to domain entity"""
        return ChatMessage(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,  # Versioning parent
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
        return cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,  # Versioning parent
            parent_message_uuid=entity.parent_message_uuid,  # Threading parent
            timestamp=entity.timestamp,
            role=entity.role.value,
            content=entity.content,
            author_uuid=entity.author_uuid,
            chat_thread_id=entity.chat_thread_id,
            tool_name=entity.tool_name,
            tool_uuid=entity.tool_uuid,
            tool_type=entity.tool_type,
            oai_tool_call_id=entity.oai_tool_call_id,
            tool_json_schema=entity.tool_json_schema,
            tool_call=entity.tool_call,
            usage=UsageSQL.from_entity(entity.usage) if entity.usage else None
        )

###############################################################################
# ToolSQL (Unifying CallableTool & StructuredTool)
###############################################################################

class ToolSQL(SQLModel, table=True):
    """SQL model for Tool entity"""
    # Primary key and versioning
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="toolsql.id")

    # Basic fields
    name: str
    
    # Tool type specific fields
    json_schema: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))  # For StructuredTool
    docstring: Optional[str] = None  # For CallableTool
    callable_text: Optional[str] = None  # For CallableTool
    input_schema: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))  # For CallableTool
    output_schema: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))  # For CallableTool
    strict_schema: bool = True  # For CallableTool

    # Relationships
    chat_thread_id: Optional[UUID] = None
    threads: List["ChatThreadSQL"] = Relationship(
        back_populates="tools",
        link_model=ThreadToolLinkSQL
    )

    def to_entity(self) -> Union[CallableTool, StructuredTool]:
        """Convert to domain entity"""
        base_args = {
            "id": self.id,
            "lineage_id": self.lineage_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "from_storage": True
        }
        
        if self.callable_text is not None:
            return CallableTool(
                **base_args,
                docstring=self.docstring,
                input_schema=self.input_schema,
                output_schema=self.output_schema,
                strict_schema=self.strict_schema,
                callable_text=self.callable_text
            )
        else:
            return StructuredTool(
                **base_args,
                description=self.docstring or "",  # Use docstring field for StructuredTool description
                json_schema=self.json_schema or {}
            )

    @classmethod
    def from_entity(cls, entity: Union[CallableTool, StructuredTool]) -> "ToolSQL":
        """Create from domain entity"""
        base_args = {
            "id": entity.id,
            "lineage_id": entity.lineage_id,
            "parent_id": entity.parent_id,
            "name": entity.name,
        }
        
        if isinstance(entity, CallableTool):
            return cls(
                **base_args,
                docstring=entity.docstring,
                input_schema=entity.input_schema,
                output_schema=entity.output_schema,
                strict_schema=entity.strict_schema,
                callable_text=entity.callable_text,
                json_schema=None
            )
        else:
            return cls(
                **base_args,
                docstring=entity.description,  # Store StructuredTool description in docstring field
                json_schema=entity.json_schema,
                callable_text=None,
                input_schema={},
                output_schema={}
            )

###############################################################################
# SystemPromptSQL
###############################################################################

class SystemPromptSQL(SQLModel, table=True):
    """SQL model for the SystemPrompt entity."""
    # Versioning
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="systempromptsql.id")

    # Domain
    name: str
    content: str

    def to_entity(self) -> SystemPrompt:
        return SystemPrompt(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            name=self.name,
            content=self.content,
            from_storage=True
        )

    @classmethod
    def from_entity(cls, entity: SystemPrompt) -> "SystemPromptSQL":
        return cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            name=entity.name,
            content=entity.content
        )

###############################################################################
# LLMConfigSQL
###############################################################################

class LLMConfigSQL(SQLModel, table=True):
    """SQL model for the LLMConfig entity."""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="llmconfigsql.id")

    client: str = Field(sa_column=Column(SQLAlchemyString))  # Store enum as string
    model: Optional[str] = None
    max_tokens: int = 400
    temperature: float = 0
    response_format: str = Field(sa_column=Column(SQLAlchemyString))  # Store enum as string
    use_cache: bool = True
    reasoner: bool = False
    reasoning_effort: Literal["low", "medium", "high"] = "medium"  # Add proper typing

    # Relationships
    threads: List["ChatThreadSQL"] = Relationship(back_populates="llm_config")

    def to_entity(self) -> LLMConfig:
        """Convert to domain entity"""
        return LLMConfig(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
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
        return cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            client=entity.client.value,  # Store enum value as string
            model=entity.model,
            max_tokens=entity.max_tokens,
            temperature=entity.temperature,
            response_format=entity.response_format.value,  # Store enum value as string
            use_cache=entity.use_cache,
            reasoner=entity.reasoner,
            reasoning_effort=entity.reasoning_effort
        )

###############################################################################
# RawOutputSQL
###############################################################################

class RawOutputSQL(SQLModel, table=True):
    """SQL model for RawOutput"""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="rawoutputsql.id")
    raw_result: Dict[str, Any] = Field(sa_column=Column(JSON))
    completion_kwargs: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    chat_thread_id: Optional[UUID] = Field(default=None, foreign_key="chatthreadsql.id")
    chat_thread_live_id: Optional[UUID] = None
    start_time: float
    end_time: float
    client: str = Field(sa_column=Column(SQLAlchemyString))

    def to_entity(self) -> RawOutput:
        return RawOutput(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
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
        return cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            raw_result=entity.raw_result,
            completion_kwargs=entity.completion_kwargs or {},
            chat_thread_id=entity.chat_thread_id,
            chat_thread_live_id=entity.chat_thread_live_id,
            start_time=entity.start_time,
            end_time=entity.end_time,
            client=entity.client.value
        )

###############################################################################
# ProcessedOutputSQL
###############################################################################

class ProcessedOutputSQL(SQLModel, table=True):
    """SQL model for the ProcessedOutput entity."""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="processedoutputsql.id")

    content: Optional[str] = None
    json_object_id: Optional[UUID] = Field(default=None, foreign_key="generatedjsonobjectsql.id")
    usage_id: Optional[UUID] = Field(default=None, foreign_key="usagesql.id")
    error: Optional[str] = None
    time_taken: float
    llm_client: str = Field(sa_column=Column(SQLAlchemyString))

    # Relationship to raw output
    raw_output_id: Optional[UUID] = Field(default=None, foreign_key="rawoutputsql.id")
    raw_output: Optional[RawOutputSQL] = Relationship(sa_relationship_kwargs={"lazy": "joined"})

    # ChatThread IDs
    chat_thread_id: UUID
    chat_thread_live_id: UUID

    # Relationships
    json_object: Optional[GeneratedJsonObjectSQL] = Relationship(
        link_model=OutputJsonObjectLinkageSQL,
        sa_relationship_kwargs={"lazy": "joined"}
    )
    usage: Optional[UsageSQL] = Relationship(sa_relationship_kwargs={"lazy": "joined"})

    def to_entity(self) -> ProcessedOutput:
        raw = self.raw_output.to_entity() if self.raw_output else None
        if not raw:
            raise ValueError("ProcessedOutput requires a RawOutput to be present in DB.")

        return ProcessedOutput(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            content=self.content,
            json_object=self.json_object.to_entity() if self.json_object else None,
            usage=self.usage.to_entity() if self.usage else None,
            error=self.error,
            time_taken=self.time_taken,
            llm_client=LLMClient(self.llm_client),
            raw_output=raw,
            chat_thread_id=self.chat_thread_id,
            chat_thread_live_id=self.chat_thread_live_id,
            from_storage=True
        )

    @classmethod
    def from_entity(cls, ent: ProcessedOutput) -> "ProcessedOutputSQL":
        return cls(
            id=ent.id,
            lineage_id=ent.lineage_id,
            parent_id=ent.parent_id,
            content=ent.content,
            error=ent.error,
            time_taken=ent.time_taken,
            llm_client=ent.llm_client.value,
            chat_thread_id=ent.chat_thread_id,
            chat_thread_live_id=ent.chat_thread_live_id,
            json_object_id=ent.json_object.id if (ent.json_object and ent.json_object.id) else None,
            usage_id=ent.usage.id if (ent.usage and ent.usage.id) else None,
            raw_output_id=ent.raw_output.id if ent.raw_output else None
        )

###############################################################################
# ChatThreadSQL
###############################################################################

class ChatThreadSQL(SQLModel, table=True):
    """SQL model for the ChatThread entity."""
    # Versioning
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="chatthreadsql.id")

    # Domain fields
    created_at: datetime = Field(default_factory=datetime.utcnow)
    name: Optional[str] = None
    new_message: Optional[str] = None
    prefill: str = "Here's the valid JSON object response:```json"
    postfill: str = "\n\nPlease provide your response in JSON format."
    use_schema_instruction: bool = True
    use_history: bool = True
    workflow_step: Optional[int] = None

    # Relationship to system prompt
    system_prompt_id: Optional[UUID] = Field(default=None, foreign_key="systempromptsql.id")
    system_prompt: Optional[SystemPromptSQL] = Relationship(sa_relationship_kwargs={"lazy": "joined"})

    # Relationship to LLMConfig
    llm_config_id: Optional[UUID] = Field(default=None, foreign_key="llmconfigsql.id")
    llm_config: Optional[LLMConfigSQL] = Relationship(
        back_populates="threads", 
        sa_relationship_kwargs={"lazy": "joined"}
    )

    # Relationship to messages (M-to-M)
    messages: List[ChatMessageSQL] = Relationship(
        back_populates="threads",
        link_model=ThreadMessageLinkSQL,
        sa_relationship_kwargs={"order_by": "ChatMessageSQL.timestamp"}
    )

    # Relationship to tools (M-to-M)
    tools: List[ToolSQL] = Relationship(
        back_populates="threads",
        link_model=ThreadToolLinkSQL
    )

    # Possibly forced_output ID (structured output in old system)
    forced_output_id: Optional[UUID] = Field(default=None, foreign_key="toolsql.id")
    forced_output: Optional[ToolSQL] = Relationship(
        sa_relationship_kwargs={"lazy": "joined"}
    )

    def to_entity(self) -> ChatThread:
        """Convert to domain entity"""
        message_entities = [m.to_entity() for m in self.messages]
        forced_output_entity = self.forced_output.to_entity() if self.forced_output else None
        system_prompt_entity = self.system_prompt.to_entity() if self.system_prompt else None
        if not self.llm_config:
            raise ValueError("ChatThread requires a LLMConfig to be present in DB.")
        llm_config_ent = self.llm_config.to_entity() 
        tool_entities = [t.to_entity() for t in self.tools]

        return ChatThread(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            name=self.name,
            system_prompt=system_prompt_entity,
            history=message_entities,
            new_message=self.new_message,
            prefill=self.prefill,
            postfill=self.postfill,
            use_schema_instruction=self.use_schema_instruction,
            use_history=self.use_history,
            forced_output=forced_output_entity,
            llm_config=llm_config_ent,
            tools=tool_entities,
            workflow_step=self.workflow_step,
            from_storage=True
        )

    @classmethod
    def from_entity(cls, ent: ChatThread) -> "ChatThreadSQL":
        """Create from domain entity"""
        return cls(
            id=ent.id,
            lineage_id=ent.lineage_id,
            parent_id=ent.parent_id,
            name=ent.name,
            new_message=ent.new_message,
            prefill=ent.prefill,
            postfill=ent.postfill,
            use_schema_instruction=ent.use_schema_instruction,
            use_history=ent.use_history,
            workflow_step=ent.workflow_step,
            system_prompt_id=ent.system_prompt.id if ent.system_prompt else None,
            llm_config_id=ent.llm_config.id if ent.llm_config else None,
            forced_output_id=ent.forced_output.id if ent.forced_output else None
        )