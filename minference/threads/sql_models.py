"""
Revised SQL models implementation using the same approach as the original working code.
Fixes chat history persistence by properly handling the many-to-many relationships.
"""

from typing import List, Optional, Dict, Any, Union, Literal, cast
from uuid import UUID, uuid4
from datetime import datetime, UTC

from sqlmodel import SQLModel, Field, Relationship, Session
from sqlalchemy import String, Column, JSON
from sqlalchemy.types import String as SQLAlchemyString
from enum import Enum
import logging

from minference.threads.models import (
    ChatThread, ChatMessage, LLMConfig, Usage, GeneratedJsonObject,
    CallableTool, StructuredTool, RawOutput, ProcessedOutput, SystemPrompt,
    MessageRole, LLMClient, ToolType, ResponseFormat
)

logger = logging.getLogger(__name__)

###############################################################################
# Type Definitions and Conversion Helpers
###############################################################################

# Define Literal types based on the enums
MessageRoleLiteral = Literal["user", "assistant", "tool", "system", "developer"]
LLMClientLiteral = Literal["openai", "azure_openai", "anthropic", "vllm", "litellm", "openrouter"]
ResponseFormatLiteral = Literal["json_beg", "text", "json_object", "structured_output", "tool", "auto_tools", "workflow"]
ToolTypeLiteral = Literal["Callable", "Structured"]
ReasoningEffortLiteral = Literal["low", "medium", "high"]

# Helper functions to convert between enums and literals
def message_role_to_literal(role: MessageRole) -> MessageRoleLiteral:
    return cast(MessageRoleLiteral, role.value)

def literal_to_message_role(role: MessageRoleLiteral) -> MessageRole:
    return MessageRole(role)

def llm_client_to_literal(client: LLMClient) -> LLMClientLiteral:
    return cast(LLMClientLiteral, client.value)

def literal_to_llm_client(client: LLMClientLiteral) -> LLMClient:
    return LLMClient(client)

def response_format_to_literal(fmt: ResponseFormat) -> ResponseFormatLiteral:
    return cast(ResponseFormatLiteral, fmt.value)

def literal_to_response_format(fmt: ResponseFormatLiteral) -> ResponseFormat:
    return ResponseFormat(fmt)

###############################################################################
# Bridging Tables (Many-to-Many)
###############################################################################

class ThreadMessageLinkSQL(SQLModel, table=True):
    """M2M link between ChatThreadSQL and ChatMessageSQL."""
    __table_args__ = {'extend_existing': True}
    
    thread_id: UUID = Field(foreign_key="chatthreadsql.id", primary_key=True)
    message_id: UUID = Field(foreign_key="chatmessagesql.id", primary_key=True)


class ThreadToolLinkSQL(SQLModel, table=True):
    """M2M link between ChatThreadSQL and ToolSQL."""
    __table_args__ = {'extend_existing': True}
    
    thread_id: UUID = Field(foreign_key="chatthreadsql.id", primary_key=True)
    tool_id: UUID = Field(foreign_key="toolsql.id", primary_key=True)


class OutputJsonObjectLinkageSQL(SQLModel, table=True):
    """Link model between ProcessedOutput and GeneratedJsonObject."""
    __table_args__ = {'extend_existing': True}
    
    processed_output_id: UUID = Field(foreign_key="processedoutputsql.id", primary_key=True)
    json_object_id: UUID = Field(foreign_key="generatedjsonobjectsql.id", primary_key=True)


###############################################################################
# Main Models
###############################################################################

class GeneratedJsonObjectSQL(SQLModel, table=True):
    """SQL model for GeneratedJsonObject."""
    __table_args__ = {'extend_existing': True}

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="generatedjsonobjectsql.id")
    name: str
    object: Dict[str, Any] = Field(sa_column=Column(JSON))
    tool_call_id: Optional[str] = None

    def to_entity(self) -> GeneratedJsonObject:
        """Convert to domain entity."""
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
        """Create from domain entity."""
        return cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            name=entity.name,
            object=entity.object,
            tool_call_id=entity.tool_call_id
        )


class UsageSQL(SQLModel, table=True):
    """SQL model for Usage."""
    __table_args__ = {'extend_existing': True}

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="usagesql.id")
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
        """Convert to domain entity."""
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
        """Create from domain entity."""
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


class ChatMessageSQL(SQLModel, table=True):
    """SQL model for ChatMessage."""
    __table_args__ = {'extend_existing': True}

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="chatmessagesql.id")
    parent_message_uuid: Optional[UUID] = Field(default=None, foreign_key="chatmessagesql.id")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    role: str = Field(sa_column=Column(SQLAlchemyString))
    content: str
    author_uuid: Optional[UUID] = None
    chat_thread_id: Optional[UUID] = None
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
        """Convert to domain entity."""
        return ChatMessage(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            parent_message_uuid=self.parent_message_uuid,
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
        """Create from domain entity."""
        return cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
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
            tool_json_schema=entity.tool_json_schema,
            tool_call=entity.tool_call,
            usage=UsageSQL.from_entity(entity.usage) if entity.usage else None
        )


class SystemPromptSQL(SQLModel, table=True):
    """SQL model for SystemPrompt."""
    __table_args__ = {'extend_existing': True}

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="systempromptsql.id")
    name: str
    content: str

    def to_entity(self) -> SystemPrompt:
        """Convert to domain entity."""
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
        """Create from domain entity."""
        return cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            name=entity.name,
            content=entity.content
        )


class ToolSQL(SQLModel, table=True):
    """SQL model for Tool entity."""
    __table_args__ = {'extend_existing': True}

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="toolsql.id")
    name: str
    json_schema: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    docstring: Optional[str] = None
    callable_text: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    output_schema: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    strict_schema: bool = True

    # Relationships
    chat_thread_id: Optional[UUID] = None
    threads: List["ChatThreadSQL"] = Relationship(
        back_populates="tools",
        link_model=ThreadToolLinkSQL
    )

    def to_entity(self) -> Union[CallableTool, StructuredTool]:
        """Convert to domain entity."""
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
                description=self.docstring or "",
                json_schema=self.json_schema or {}
            )

    @classmethod
    def from_entity(cls, entity: Union[CallableTool, StructuredTool]) -> "ToolSQL":
        """Create from domain entity."""
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
                docstring=entity.description,
                json_schema=entity.json_schema,
                callable_text=None,
                input_schema={},
                output_schema={}
            )


class LLMConfigSQL(SQLModel, table=True):
    """SQL model for LLMConfig."""
    __table_args__ = {'extend_existing': True}

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="llmconfigsql.id")
    client: str = Field(sa_column=Column(SQLAlchemyString))
    model: Optional[str] = None
    max_tokens: int = 400
    temperature: float = 0
    response_format: str = Field(sa_column=Column(SQLAlchemyString))
    use_cache: bool = True

    # Relationships
    threads: List["ChatThreadSQL"] = Relationship(back_populates="llm_config")

    def to_entity(self) -> LLMConfig:
        """Convert to domain entity."""
        return LLMConfig(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            client=LLMClient(self.client),
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format=ResponseFormat(self.response_format),
            use_cache=self.use_cache,
            from_storage=True
        )

    @classmethod
    def from_entity(cls, entity: LLMConfig) -> "LLMConfigSQL":
        """Create from domain entity."""
        return cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            client=entity.client.value,
            model=entity.model,
            max_tokens=entity.max_tokens,
            temperature=entity.temperature,
            response_format=entity.response_format.value,
            use_cache=entity.use_cache
        )


class RawOutputSQL(SQLModel, table=True):
    """SQL model for RawOutput."""
    __table_args__ = {'extend_existing': True}

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="rawoutputsql.id")
    raw_result: Dict[str, Any] = Field(sa_column=Column(JSON))
    completion_kwargs: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    chat_thread_id: Optional[UUID] = Field(default=None, foreign_key="chatthreadsql.id")
    start_time: float
    end_time: float
    client: str = Field(sa_column=Column(SQLAlchemyString))

    def to_entity(self) -> RawOutput:
        """Convert to domain entity."""
        return RawOutput(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            raw_result=self.raw_result,
            completion_kwargs=self.completion_kwargs,
            chat_thread_id=self.chat_thread_id,
            start_time=self.start_time,
            end_time=self.end_time,
            client=LLMClient(self.client),
            from_storage=True
        )

    @classmethod
    def from_entity(cls, entity: RawOutput) -> "RawOutputSQL":
        """Create from domain entity."""
        return cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            raw_result=entity.raw_result,
            completion_kwargs=entity.completion_kwargs or {},
            chat_thread_id=entity.chat_thread_id,
            start_time=entity.start_time,
            end_time=entity.end_time,
            client=entity.client.value
        )


class ProcessedOutputSQL(SQLModel, table=True):
    """SQL model for ProcessedOutput."""
    __table_args__ = {'extend_existing': True}

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="processedoutputsql.id")
    content: Optional[str] = None
    error: Optional[str] = None
    time_taken: float
    llm_client: str = Field(sa_column=Column(SQLAlchemyString))
    
    # Foreign keys
    json_object_id: Optional[UUID] = Field(default=None, foreign_key="generatedjsonobjectsql.id")
    usage_id: Optional[UUID] = Field(default=None, foreign_key="usagesql.id")
    raw_output_id: Optional[UUID] = Field(default=None, foreign_key="rawoutputsql.id")
    chat_thread_id: UUID
    chat_thread_live_id: UUID

    # Relationships
    json_object: Optional[GeneratedJsonObjectSQL] = Relationship(sa_relationship_kwargs={"lazy": "joined"})
    usage: Optional[UsageSQL] = Relationship(sa_relationship_kwargs={"lazy": "joined"})
    raw_output: Optional[RawOutputSQL] = Relationship(sa_relationship_kwargs={"lazy": "joined"})

    def to_entity(self) -> ProcessedOutput:
        """Convert to domain entity."""
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
        """Create from domain entity."""
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


class ChatThreadSQL(SQLModel, table=True):
    """SQL model for ChatThread."""
    __table_args__ = {'extend_existing': True}

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="chatthreadsql.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    name: Optional[str] = None
    new_message: Optional[str] = None
    prefill: str = "Here's the valid JSON object response:```json"
    postfill: str = "\n\nPlease provide your response in JSON format."
    use_schema_instruction: bool = True
    use_history: bool = True
    workflow_step: Optional[int] = None

    # Foreign keys
    system_prompt_id: Optional[UUID] = Field(default=None, foreign_key="systempromptsql.id")
    llm_config_id: Optional[UUID] = Field(default=None, foreign_key="llmconfigsql.id")
    forced_output_id: Optional[UUID] = Field(default=None, foreign_key="toolsql.id")

    # Relationships
    system_prompt: Optional[SystemPromptSQL] = Relationship(sa_relationship_kwargs={"lazy": "joined"})
    llm_config: Optional[LLMConfigSQL] = Relationship(back_populates="threads", sa_relationship_kwargs={"lazy": "joined"})
    forced_output: Optional[ToolSQL] = Relationship(sa_relationship_kwargs={"lazy": "joined"})
    messages: List[ChatMessageSQL] = Relationship(
        back_populates="threads",
        link_model=ThreadMessageLinkSQL,
        sa_relationship_kwargs={"lazy": "joined"}
    )
    tools: List[ToolSQL] = Relationship(
        back_populates="threads",
        link_model=ThreadToolLinkSQL,
        sa_relationship_kwargs={"lazy": "joined"}
    )

    def to_entity(self) -> ChatThread:
        """Convert to domain entity."""
        message_entities = [m.to_entity() for m in self.messages]
        forced_output_entity = self.forced_output.to_entity() if self.forced_output else None
        system_prompt_entity = self.system_prompt.to_entity() if self.system_prompt else None

        if not self.llm_config:
            raise ValueError("ChatThread requires an LLMConfig to be present in DB.")
        
        llm_config_ent = self.llm_config.to_entity()
        tool_entities = [t.to_entity() for t in self.tools]

        logger.debug(f"Converting ChatThreadSQL({self.id}) to entity with {len(message_entities)} messages")

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
        """Create SQL model from domain entity (basic fields only)."""
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


###############################################################################
# SqlEntityStorageHelper 
###############################################################################

class SqlEntityStorageHelper:
    """Helper class for SQL entity storage operations."""

    def __init__(self, session: Session):
        self.session = session

    def save_thread(self, thread_ent: ChatThread) -> None:
        """Save a ChatThread entity and all its related entities."""
        # 1) Convert to ChatThreadSQL
        thread_sql = ChatThreadSQL.from_entity(thread_ent)

        # 2) Upsert system_prompt, llm_config, forced_output, etc.
        if thread_ent.system_prompt:
            sp_sql = SystemPromptSQL.from_entity(thread_ent.system_prompt)
            self.session.merge(sp_sql)  # or add if new

        if thread_ent.llm_config:
            config_sql = LLMConfigSQL.from_entity(thread_ent.llm_config)
            self.session.merge(config_sql)

        if thread_ent.forced_output:
            forced_sql = ToolSQL.from_entity(thread_ent.forced_output)
            self.session.merge(forced_sql)

        # 3) For each message in thread_ent.history, upsert
        message_sql_list: List[ChatMessageSQL] = []
        for msg_ent in thread_ent.history:
            msg_sql = ChatMessageSQL.from_entity(msg_ent)
            self.session.merge(msg_sql)
            message_sql_list.append(msg_sql)

        # 4) For each tool in thread_ent.tools, upsert
        tool_sql_list: List[ToolSQL] = []
        for tool_ent in thread_ent.tools:
            tool_sql = ToolSQL.from_entity(tool_ent)
            self.session.merge(tool_sql)
            tool_sql_list.append(tool_sql)

        # 5) Merge the ChatThreadSQL row itself
        self.session.merge(thread_sql)
        self.session.commit()

        # 6) Now that we have IDs, attach message -> thread bridging
        thread_sql_db = self.session.get(ChatThreadSQL, thread_sql.id)
        if not thread_sql_db:
            raise ValueError(f"Failed to retrieve ChatThreadSQL after merge (ID={thread_sql.id})")

        # Attach messages
        for msg_sql in message_sql_list:
            if msg_sql not in thread_sql_db.messages:
                thread_sql_db.messages.append(msg_sql)

        # Attach tools
        for tool_sql in tool_sql_list:
            if tool_sql not in thread_sql_db.tools:
                thread_sql_db.tools.append(tool_sql)

        self.session.add(thread_sql_db)
        self.session.commit()

    def load_thread(self, thread_id: UUID) -> ChatThread:
        """Load a ChatThread entity from DB by its version ID (UUID)."""
        thread_sql = self.session.get(ChatThreadSQL, thread_id)
        if not thread_sql:
            raise ValueError(f"No ChatThreadSQL found with ID={thread_id}")
        return thread_sql.to_entity()