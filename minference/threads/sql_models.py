"""
Fully refactored sql_models.py illustrating:
 - UUID-based primary keys
 - parent_id / lineage_id for versioning
 - bridging tables for many-to-many relationships
 - basic from_entity() / to_entity() methods

NOTE: This code is a reference skeleton, not drop-in ready.
You would adapt queries, session usage, etc. to your application.
It has been adjusted to avoid linting issues (e.g., avoid calling Literal[...] as a constructor).
"""

from typing import List, Optional, Dict, Any, Union, Literal, cast
from uuid import UUID, uuid4
from datetime import datetime

from sqlmodel import SQLModel, Field, Relationship, Session
from enum import Enum

from minference.threads.models import (
    ChatThread, ChatMessage, LLMConfig, Usage, GeneratedJsonObject,
    CallableTool, StructuredTool, RawOutput, ProcessedOutput, SystemPrompt,
    MessageRole, LLMClient, ToolType, ResponseFormat
)

###############################################################################
# Bridging Tables (Many-to-Many)
###############################################################################


class ThreadMessageLinkSQL(SQLModel, table=True):
    """
    Many-to-many link between ChatThreadSQL and ChatMessageSQL.
    Allows multiple threads to reference the same message version,
    enabling one-to-many or many-to-many usage.
    """
    __table_args__ = {'extend_existing': True}
    thread_id: UUID = Field(foreign_key="chatthreadsql.id", primary_key=True)
    message_id: UUID = Field(foreign_key="chatmessagesql.id", primary_key=True)


class ThreadToolLinkSQL(SQLModel, table=True):
    """
    Many-to-many link between ChatThreadSQL and ToolSQL.
    Because ChatThread.tools is List[Union[CallableTool, StructuredTool]],
    we unify them in a single ToolSQL table with a type discriminator.
    """
    __table_args__ = {'extend_existing': True}
    thread_id: UUID = Field(foreign_key="chatthreadsql.id", primary_key=True)
    tool_id: UUID = Field(foreign_key="toolsql.id", primary_key=True)


###############################################################################
# ChatMessageSQL
###############################################################################


class ChatMessageSQL(SQLModel, table=True):
    """
    SQL model for a single ChatMessage entity version.
    """
    __table_args__ = {'extend_existing': True}

    # Versioning
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = Field(default=None, foreign_key="chatmessagesql.id")

    # Domain fields
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    role: MessageRole
    content: str
    author_uuid: Optional[UUID] = None
    chat_thread_id: Optional[UUID] = None  # Soft link to a single thread if desired
    parent_message_uuid: Optional[UUID] = None

    # Tool usage
    tool_name: Optional[str] = None
    tool_uuid: Optional[UUID] = None
    tool_type: Optional[ToolType] = None
    oai_tool_call_id: Optional[str] = None
    tool_json_schema: Optional[Dict[str, Any]] = None
    tool_call: Optional[Dict[str, Any]] = None

    # Usage stats inline for assistant messages
    usage_id: Optional[UUID] = Field(default=None, foreign_key="usagesql.id")

    # Relationship back to usage row
    usage: Optional["UsageSQL"] = Relationship(sa_relationship_kwargs={"lazy": "joined"})

    # Relationship to threads (M-to-M)
    threads: List["ChatThreadSQL"] = Relationship(
        back_populates="messages",
        link_model=ThreadMessageLinkSQL
    )

    ###########################################################################
    # Mappers to the Domain "ChatMessage" entity
    ###########################################################################
    def to_entity(self) -> ChatMessage:
        """
        Convert this row to a ChatMessage domain entity (Pydantic).
        NOTE: ChatMessage is from the code in models.py
        """
        # We convert tool_type from a Union[Literal["Callable","Structured"]] to the same literal in domain
        domain_tool_type = cast(ToolType, self.tool_type) if self.tool_type else None

        return ChatMessage(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            timestamp=self.timestamp,
            role=self.role,
            content=self.content,
            author_uuid=self.author_uuid,
            chat_thread_id=self.chat_thread_id,
            parent_message_uuid=self.parent_message_uuid,
            tool_name=self.tool_name,
            tool_uuid=self.tool_uuid,
            tool_type=domain_tool_type,
            oai_tool_call_id=self.oai_tool_call_id,
            tool_json_schema=self.tool_json_schema,
            tool_call=self.tool_call,
            usage=self.usage.to_entity() if self.usage else None
        )

    @classmethod
    def from_entity(cls, entity: ChatMessage) -> "ChatMessageSQL":
        """
        Build a ChatMessageSQL row from a ChatMessage entity.
        """
        return cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            timestamp=entity.timestamp,
            role=entity.role,
            content=entity.content,
            author_uuid=entity.author_uuid,
            chat_thread_id=entity.chat_thread_id,
            parent_message_uuid=entity.parent_message_uuid,
            tool_name=entity.tool_name,
            tool_uuid=entity.tool_uuid,
            tool_type=entity.tool_type,
            oai_tool_call_id=entity.oai_tool_call_id,
            tool_json_schema=entity.tool_json_schema,
            tool_call=entity.tool_call,
            usage_id=entity.usage.id if (entity.usage and entity.usage.id) else None
        )


###############################################################################
# UsageSQL
###############################################################################


class UsageSQL(SQLModel, table=True):
    """
    Tracks usage. The domain is "Usage" in models.py
    """
    __table_args__ = {'extend_existing': True}

    # Versioning
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = None

    # Domain fields
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    accepted_prediction_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None

    # Relationship back to messages that reference usage
    message: List[ChatMessageSQL] = Relationship(sa_relationship_kwargs={"lazy": "joined"})

    ###########################################################################
    # Mappers
    ###########################################################################
    def to_entity(self) -> Usage:
        return Usage(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            model=self.model,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens,
            accepted_prediction_tokens=self.accepted_prediction_tokens,
            audio_tokens=self.audio_tokens,
            reasoning_tokens=self.reasoning_tokens,
            rejected_prediction_tokens=self.rejected_prediction_tokens,
            cached_tokens=self.cached_tokens
        )

    @classmethod
    def from_entity(cls, entity: Usage) -> "UsageSQL":
        return cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            model=entity.model,
            prompt_tokens=entity.prompt_tokens,
            completion_tokens=entity.completion_tokens,
            total_tokens=entity.total_tokens,
            cache_creation_input_tokens=entity.cache_creation_input_tokens,
            cache_read_input_tokens=entity.cache_read_input_tokens,
            accepted_prediction_tokens=entity.accepted_prediction_tokens,
            audio_tokens=entity.audio_tokens,
            reasoning_tokens=entity.reasoning_tokens,
            rejected_prediction_tokens=entity.rejected_prediction_tokens,
            cached_tokens=entity.cached_tokens
        )


###############################################################################
# ToolSQL (Unifying CallableTool & StructuredTool)
###############################################################################


class ToolTypeEnum(str, Enum):
    callable = "callable"
    structured = "structured"


class ToolSQL(SQLModel, table=True):
    """
    Single table for both CallableTool and StructuredTool.
    We'll store a 'tool_type' field to differentiate them.
    """
    __table_args__ = {'extend_existing': True}

    # Versioning
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = None

    # A discriminator
    tool_type: ToolTypeEnum

    # Fields for CallableTool
    name: str
    docstring: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    strict_schema: bool = True
    callable_text: Optional[str] = None

    # Fields for StructuredTool
    description: Optional[str] = None
    instruction_string: Optional[str] = None
    json_schema: Optional[Dict[str, Any]] = None

    # Relationship: which ChatThreads reference this tool
    threads: List["ChatThreadSQL"] = Relationship(
        back_populates="tools",
        link_model=ThreadToolLinkSQL
    )

    ###########################################################################
    # Mappers
    ###########################################################################
    def to_entity(self) -> Union[CallableTool, StructuredTool]:
        """Convert this row to either a CallableTool or StructuredTool entity."""
        if self.tool_type == ToolTypeEnum.callable:
            return CallableTool(
                id=self.id,
                lineage_id=self.lineage_id,
                parent_id=self.parent_id,
                name=self.name,
                docstring=self.docstring,
                input_schema=self.input_schema or {},
                output_schema=self.output_schema or {},
                strict_schema=self.strict_schema,
                callable_text=self.callable_text
            )
        else:
            return StructuredTool(
                id=self.id,
                lineage_id=self.lineage_id,
                parent_id=self.parent_id,
                name=self.name,
                description=self.description or "",
                instruction_string=self.instruction_string or "",
                json_schema=self.json_schema or {},
                strict_schema=self.strict_schema
            )

    @classmethod
    def from_entity(cls, entity: Union[CallableTool, StructuredTool]) -> "ToolSQL":
        is_callable = isinstance(entity, CallableTool)
        tool_type = ToolTypeEnum.callable if is_callable else ToolTypeEnum.structured

        base_fields = {
            "id": entity.id,
            "lineage_id": entity.lineage_id,
            "parent_id": entity.parent_id,
            "tool_type": tool_type,
            "name": entity.name,
            "strict_schema": entity.strict_schema
        }

        if is_callable:
            return cls(
                **base_fields,
                docstring=entity.docstring,
                input_schema=entity.input_schema,
                output_schema=entity.output_schema,
                callable_text=entity.callable_text
            )
        else:
            return cls(
                **base_fields,
                description=getattr(entity, "description", None),
                instruction_string=getattr(entity, "instruction_string", None),
                json_schema=getattr(entity, "json_schema", None)
            )


###############################################################################
# SystemPromptSQL
###############################################################################


class SystemPromptSQL(SQLModel, table=True):
    """
    For the SystemPrompt entity. Possibly many threads can reference
    the same system prompt version, or treat it as 1-to-1.
    """
    __table_args__ = {'extend_existing': True}

    # Versioning
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = None

    # Domain
    name: str
    content: str

    ###########################################################################
    # Mappers
    ###########################################################################
    def to_entity(self) -> SystemPrompt:
        return SystemPrompt(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            name=self.name,
            content=self.content
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
    """
    For the LLMConfig entity. Some chat threads reference a single config.
    """
    __table_args__ = {'extend_existing': True}

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = None

    client: LLMClient
    model: Optional[str] = None
    max_tokens: int = 400
    temperature: float = 0.0
    response_format: str = "text"
    use_cache: bool = True
    reasoner: bool = False
    reasoning_effort: Literal["low", "medium", "high"] = "medium"

    ###########################################################################
    # Mappers
    ###########################################################################
    def to_entity(self) -> LLMConfig:
        return LLMConfig(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            client=self.client,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format=ResponseFormat(self.response_format),
            use_cache=self.use_cache,
            reasoner=self.reasoner,
            reasoning_effort=self.reasoning_effort
        )

    @classmethod
    def from_entity(cls, entity: LLMConfig) -> "LLMConfigSQL":
        return cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            client=entity.client,
            model=entity.model,
            max_tokens=entity.max_tokens,
            temperature=entity.temperature,
            response_format=entity.response_format.value,  # convert enum to str
            use_cache=entity.use_cache,
            reasoner=entity.reasoner,
            reasoning_effort=entity.reasoning_effort
        )


###############################################################################
# GeneratedJsonObjectSQL
###############################################################################


class GeneratedJsonObjectSQL(SQLModel, table=True):
    """
    For the GeneratedJsonObject entity.
    """
    __table_args__ = {'extend_existing': True}

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = None

    name: str
    object: Dict[str, Any] = Field(default_factory=dict)
    tool_call_id: Optional[str] = None

    ###########################################################################
    # Mappers
    ###########################################################################
    def to_entity(self) -> GeneratedJsonObject:
        return GeneratedJsonObject(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            name=self.name,
            object=self.object,
            tool_call_id=self.tool_call_id
        )

    @classmethod
    def from_entity(cls, ent: GeneratedJsonObject) -> "GeneratedJsonObjectSQL":
        return cls(
            id=ent.id,
            lineage_id=ent.lineage_id,
            parent_id=ent.parent_id,
            name=ent.name,
            object=ent.object,
            tool_call_id=ent.tool_call_id
        )


###############################################################################
# RawOutputSQL
###############################################################################


class RawOutputSQL(SQLModel, table=True):
    """
    For the RawOutput entity, which has references to the chat_thread_id, etc.
    """
    __table_args__ = {'extend_existing': True}

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = None

    raw_result: Dict[str, Any] = Field(default_factory=dict)
    completion_kwargs: Dict[str, Any] = Field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    chat_thread_id: Optional[UUID] = None
    chat_thread_live_id: Optional[UUID] = None
    client: LLMClient

    ###########################################################################
    # Mappers
    ###########################################################################
    def to_entity(self) -> RawOutput:
        return RawOutput(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            raw_result=self.raw_result,
            completion_kwargs=self.completion_kwargs,
            start_time=self.start_time,
            end_time=self.end_time,
            chat_thread_id=self.chat_thread_id,
            chat_thread_live_id=self.chat_thread_live_id,
            client=self.client
        )

    @classmethod
    def from_entity(cls, ent: RawOutput) -> "RawOutputSQL":
        return cls(
            id=ent.id,
            lineage_id=ent.lineage_id,
            parent_id=ent.parent_id,
            raw_result=ent.raw_result,
            completion_kwargs=ent.completion_kwargs,
            start_time=ent.start_time,
            end_time=ent.end_time,
            chat_thread_id=ent.chat_thread_id,
            chat_thread_live_id=ent.chat_thread_live_id,
            client=ent.client
        )


###############################################################################
# ProcessedOutputSQL
###############################################################################


class ProcessedOutputSQL(SQLModel, table=True):
    """
    For the ProcessedOutput entity, referencing a RawOutput, usage, etc.
    """
    __table_args__ = {'extend_existing': True}

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = Field(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = None

    content: Optional[str] = None
    json_object_id: Optional[UUID] = Field(default=None, foreign_key="generatedjsonobjectsql.id")
    usage_id: Optional[UUID] = Field(default=None, foreign_key="usagesql.id")
    error: Optional[str] = None
    time_taken: float
    llm_client: LLMClient

    # Relationship to raw output
    raw_output_id: Optional[UUID] = Field(default=None, foreign_key="rawoutputsql.id")
    raw_output: Optional[RawOutputSQL] = Relationship(sa_relationship_kwargs={"lazy": "joined"})

    # ChatThread ID
    chat_thread_id: UUID
    chat_thread_live_id: UUID

    # Relationship
    json_object: Optional[GeneratedJsonObjectSQL] = Relationship(sa_relationship_kwargs={"lazy": "joined"})
    usage: Optional[UsageSQL] = Relationship(sa_relationship_kwargs={"lazy": "joined"})

    ###########################################################################
    # Mappers
    ###########################################################################
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
            llm_client=self.llm_client,
            raw_output=raw,
            chat_thread_id=self.chat_thread_id,
            chat_thread_live_id=self.chat_thread_live_id
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
            llm_client=ent.llm_client,
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
    """
    For the ChatThread entity. Uses many-to-many linking with messages and tools.
    """
    __table_args__ = {'extend_existing': True}

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
    llm_config: Optional[LLMConfigSQL] = Relationship(sa_relationship_kwargs={"lazy": "joined"})

    # Relationship to messages (M-to-M)
    messages: List[ChatMessageSQL] = Relationship(
        back_populates="threads",
        link_model=ThreadMessageLinkSQL
    )

    # Relationship to tools (M-to-M)
    tools: List[ToolSQL] = Relationship(
        back_populates="threads",
        link_model=ThreadToolLinkSQL
    )

    # Possibly forced_output ID
    forced_output_id: Optional[UUID] = Field(default=None, foreign_key="toolsql.id")
    forced_output: Optional[ToolSQL] = Relationship(
        sa_relationship_kwargs={"lazy": "joined"},
        back_populates=None
    )

    ###########################################################################
    # Mappers
    ###########################################################################
    def to_entity(self) -> ChatThread:
        """
        Convert ChatThreadSQL -> ChatThread domain entity.
        We also build the list of ChatMessage entities from self.messages,
        the system_prompt, the tools, etc.
        """
        message_entities = [m.to_entity() for m in self.messages]
        forced_output_entity = self.forced_output.to_entity() if self.forced_output else None
        system_prompt_entity = self.system_prompt.to_entity() if self.system_prompt else None

        if not self.llm_config:
            raise ValueError("ChatThread requires an LLMConfig to be present in DB.")
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
            workflow_step=self.workflow_step
        )

    @classmethod
    def from_entity(cls, ent: ChatThread) -> "ChatThreadSQL":
        """
        Convert ChatThread entity -> ChatThreadSQL row.
        We'll NOT automatically add messages or tools here; 
        that typically belongs in a storage manager that merges them individually.
        """
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
# Example: Storage Helper (Optional)
###############################################################################

class SqlEntityStorageHelper:
    """
    Example helper class that shows how you might insert or retrieve 
    a ChatThread from the DB using the from_entity()/to_entity() approach.
    """

    def __init__(self, session: Session):
        self.session = session

    def save_thread(self, thread_ent: ChatThread) -> None:
        """
        Insert a new version of ChatThread + related messages/tools/etc.
        If messages or tools exist in DB, we reuse them; otherwise, we create new rows.
        """
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
        """
        Load a ChatThread entity from DB by its version ID (UUID).
        """
        thread_sql = self.session.get(ChatThreadSQL, thread_id)
        if not thread_sql:
            raise ValueError(f"No ChatThreadSQL found with ID={thread_id}")
        return thread_sql.to_entity()
