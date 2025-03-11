"""
SQLAlchemy models for the Thread system.

This module provides SQLAlchemy ORM models that mirror the Pydantic models in models.py.
"""

from __future__ import annotations

import uuid
import inspect
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, cast, Type, Tuple
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, JSON, 
    String, Table, Text, Uuid, create_engine, inspect, Float
)
from sqlalchemy.orm import (
    Mapped, joinedload, mapped_column, relationship, sessionmaker, declarative_base,
    Session
)
from sqlalchemy.sql import func

# Import entity classes
from minference.threads.models import (
    ChatThread, ChatMessage, SystemPrompt, LLMConfig, 
    CallableTool, StructuredTool, Usage, GeneratedJsonObject,
    RawOutput, ProcessedOutput, LLMClient, ResponseFormat, MessageRole
)
from minference.ecs.entity import Entity
from minference.threads.inference import RequestLimits

# Create SQLAlchemy Base
Base = declarative_base()

class EntityBase(Base):
    """Abstract base class for all entity tables with common columns for versioning."""
    __abstract__ = True

    # Primary database key (auto-incremented integer)
    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Entity versioning fields from the Entity class
    ecs_id = mapped_column(Uuid, nullable=False, index=True, unique=True)
    lineage_id = mapped_column(Uuid, nullable=False, index=True)
    parent_id = mapped_column(Uuid, nullable=True, index=True)
    created_at = mapped_column(DateTime(timezone=True), nullable=False)
    old_ids = mapped_column(JSON, nullable=False, default=list)
    
    # The entity type for polymorphic identity
    entity_type = mapped_column(String(50), nullable=False)
    
    __mapper_args__ = {
        "polymorphic_on": entity_type,
    }

# Association table for ChatThread-CallableTool many-to-many relationship
chat_thread_tools = Table(
    "chat_thread_tools",
    Base.metadata,
    Column("chat_thread_id", Uuid, ForeignKey("chat_thread.ecs_id"), primary_key=True),
    Column("tool_id", Uuid, ForeignKey("tool.ecs_id"), primary_key=True),
)

# Add this class after the Base definition but before other entity classes
class BaseEntitySQL(EntityBase):
    """SQLAlchemy model for generic entity storage."""
    __tablename__ = "base_entity"
    
    # Additional fields needed for storing any entity
    entity_class = mapped_column(String(100), nullable=False)
    data = mapped_column(JSON, nullable=True)
    
    __mapper_args__ = {
        "polymorphic_identity": "base_entity",
    }
    
    @classmethod
    def from_entity(cls, entity: Entity) -> 'BaseEntitySQL':
        """Convert from Entity to SQL model."""
        # Convert UUID objects to strings for JSON serialization
        str_old_ids = [str(uid) for uid in entity.old_ids] if entity.old_ids else []
        
        return cls(
            ecs_id=entity.ecs_id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            created_at=entity.created_at,
            old_ids=str_old_ids,
            entity_class=f"{entity.__class__.__module__}.{entity.__class__.__name__}",
            data=entity.entity_dump(),
            entity_type="base_entity"
        )
    
    def to_entity(self) -> Entity:
        """Convert from SQL model to Entity."""
        # Import dynamically to avoid circular imports
        from minference.ecs.entity import dynamic_import
        
        # Get the entity class
        entity_class = dynamic_import(self.entity_class)
        
        # Create the entity from stored data
        return entity_class.model_validate({
            "ecs_id": self.ecs_id,
            "lineage_id": self.lineage_id,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "old_ids": [UUID(uid) for uid in self.old_ids] if self.old_ids else [],
            "from_storage": True,
            **self.data
        })

class ChatThreadSQL(EntityBase):
    """SQLAlchemy model for ChatThread entities."""
    __tablename__ = "chat_thread"
    
    # Relationships
    messages = relationship("ChatMessageSQL", back_populates="chat_thread")
    system_prompt_id = mapped_column(Uuid, ForeignKey("system_prompt.ecs_id"), nullable=True)
    system_prompt = relationship("SystemPromptSQL")
    llm_config_id = mapped_column(Uuid, ForeignKey("llm_config.ecs_id"), nullable=True)
    llm_config = relationship("LLMConfigSQL")
    tools = relationship("ToolSQL", secondary=chat_thread_tools)
    
    # ChatThread specific fields
    title = mapped_column(String(255), nullable=True)  # maps to 'name' in ChatThread entity
    thread_metadata = mapped_column(JSON, nullable=True)  # Renamed from 'metadata' to avoid SQLAlchemy reserved name
    workflow_step = mapped_column(Integer, nullable=True)  # Added workflow_step field
    
    # Added missing fields
    new_message = mapped_column(Text, nullable=True)
    prefill = mapped_column(Text, nullable=True, default="Here's the valid JSON object response:```json")
    postfill = mapped_column(Text, nullable=True, default="\n\nPlease provide your response in JSON format.")
    use_schema_instruction = mapped_column(Boolean, nullable=False, default=False)
    use_history = mapped_column(Boolean, nullable=False, default=True)
    
    # Forced output entity reference
    forced_output_id = mapped_column(Uuid, ForeignKey("tool.ecs_id"), nullable=True)
    forced_output = relationship("ToolSQL")
    
    __mapper_args__ = {
        "polymorphic_identity": "chat_thread",
    }
    
    def to_entity(self) -> ChatThread:
        """Convert from SQL model to Entity."""
        # Convert related entities if available
        system_prompt = self.system_prompt.to_entity() if self.system_prompt else None
        llm_config = self.llm_config.to_entity() if self.llm_config else None
        tools = [tool.to_entity() for tool in self.tools] if self.tools else []
        forced_output = self.forced_output.to_entity() if self.forced_output else None
        
        # Make sure we have a valid LLMConfig (required)
        if llm_config is None:
            # Create a minimal default LLMConfig
            from minference.threads.models import LLMConfig, LLMClient, ResponseFormat
            llm_config = LLMConfig(client=LLMClient.openai, model="gpt-4")
        
        # Convert messages if available
        history = []
        if self.messages:
            history = [msg.to_entity() for msg in self.messages]
        
        # Convert old_ids strings back to UUID objects
        uuid_old_ids = []
        if self.old_ids:
            for old_id in self.old_ids:
                if isinstance(old_id, str):
                    uuid_old_ids.append(UUID(old_id))
                elif isinstance(old_id, UUID):
                    uuid_old_ids.append(old_id)
            
        return ChatThread(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=uuid_old_ids,  # Use converted UUID objects
            name=self.title,
            system_prompt=system_prompt,
            llm_config=llm_config,
            tools=tools,
            history=history,  # Add the converted messages
            workflow_step=self.workflow_step,
            new_message=self.new_message,
            prefill=self.prefill,
            postfill=self.postfill,
            use_schema_instruction=self.use_schema_instruction,
            use_history=self.use_history,
            forced_output=forced_output,
            from_storage=True
        )
    
    @classmethod
    def from_entity(cls, entity: ChatThread) -> 'ChatThreadSQL':
        """Convert from Entity to SQL model."""
        # Convert UUID objects to strings for JSON serialization
        str_old_ids = [str(uid) for uid in entity.old_ids] if entity.old_ids else []
        
        return cls(
            ecs_id=entity.ecs_id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            created_at=entity.created_at,
            old_ids=str_old_ids,  # Use string representation for JSON serialization
            title=entity.name,
            workflow_step=entity.workflow_step,
            new_message=entity.new_message,
            prefill=entity.prefill,
            postfill=entity.postfill,
            use_schema_instruction=entity.use_schema_instruction,
            use_history=entity.use_history,
            forced_output_id=entity.forced_output.ecs_id if entity.forced_output else None,
            entity_type="chat_thread"  # Required field
        )
        
    def handle_relationships(self, entity: ChatThread, session: Session, orm_objects: Dict[UUID, Any]) -> None:
        """Handle relationships for ChatThread."""
        # Handle system_prompt relationship
        if entity.system_prompt:
            if entity.system_prompt.ecs_id in orm_objects:
                self.system_prompt = orm_objects[entity.system_prompt.ecs_id]
            else:
                # Try to find in database or create new
                system_prompt = session.query(SystemPromptSQL).filter(
                    SystemPromptSQL.ecs_id == entity.system_prompt.ecs_id
                ).first()
                if system_prompt:
                    self.system_prompt = system_prompt
                    
        # Handle llm_config relationship
        if entity.llm_config:
            if entity.llm_config.ecs_id in orm_objects:
                self.llm_config = orm_objects[entity.llm_config.ecs_id]
            else:
                # Try to find in database or create new
                llm_config = session.query(LLMConfigSQL).filter(
                    LLMConfigSQL.ecs_id == entity.llm_config.ecs_id
                ).first()
                if llm_config:
                    self.llm_config = llm_config
                    
        # Handle forced_output relationship
        if entity.forced_output:
            if entity.forced_output.ecs_id in orm_objects:
                self.forced_output = orm_objects[entity.forced_output.ecs_id]
            else:
                # Try to find in database
                forced_output = session.query(ToolSQL).filter(
                    ToolSQL.ecs_id == entity.forced_output.ecs_id
                ).first()
                if forced_output:
                    self.forced_output = forced_output
                    
        # Handle tools relationship (many-to-many)
        if entity.tools:
            tools = []
            for tool in entity.tools:
                if tool.ecs_id in orm_objects:
                    tools.append(orm_objects[tool.ecs_id])
                else:
                    # Try to find in database
                    tool_orm = session.query(ToolSQL).filter(
                        ToolSQL.ecs_id == tool.ecs_id
                    ).first()
                    if tool_orm:
                        tools.append(tool_orm)
            self.tools = tools

class ChatMessageSQL(EntityBase):
    """SQLAlchemy model for ChatMessage entities."""
    __tablename__ = "chat_message"
    
    # Relationships
    chat_thread_id = mapped_column(Uuid, ForeignKey("chat_thread.ecs_id"), nullable=True)
    chat_thread = relationship("ChatThreadSQL", back_populates="messages")
    parent_message_id = mapped_column(Uuid, ForeignKey("chat_message.ecs_id"), nullable=True)
    parent_message = relationship("ChatMessageSQL", remote_side="ChatMessageSQL.ecs_id", 
                                back_populates="child_messages")
    child_messages = relationship("ChatMessageSQL", back_populates="parent_message", 
                                overlaps="parent_message")
    tool_id = mapped_column(Uuid, ForeignKey("tool.ecs_id"), nullable=True)
    tool = relationship("ToolSQL")
    usage_id = mapped_column(Uuid, ForeignKey("usage.ecs_id"), nullable=True)
    usage = relationship("UsageSQL")
    
    # ChatMessage specific fields
    role = mapped_column(String(50), nullable=False)
    content = mapped_column(Text, nullable=True)
    author_uuid = mapped_column(Uuid, nullable=True)
    timestamp = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Tool-related fields
    tool_name = mapped_column(String(255), nullable=True)
    tool_uuid = mapped_column(Uuid, nullable=True)
    tool_type = mapped_column(String(50), nullable=True)
    oai_tool_call_id = mapped_column(String(255), nullable=True)
    tool_json_schema = mapped_column(JSON, nullable=True)
    tool_call = mapped_column(JSON, nullable=True)
    
    __mapper_args__ = {
        "polymorphic_identity": "chat_message",
    }
    
    def to_entity(self) -> ChatMessage:
        """Convert from SQL model to Entity."""
        # Load related entities if available
        usage_entity = self.usage.to_entity() if self.usage else None
        
        # Convert role string to enum
        from minference.threads.models import MessageRole
        role_value = MessageRole(self.role) if self.role else MessageRole.user
        
        # Create the entity
        return ChatMessage(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=self.old_ids,
            timestamp=self.timestamp or datetime.now(timezone.utc),
            role=role_value,
            content=self.content or "",
            author_uuid=self.author_uuid,
            chat_thread_id=self.chat_thread_id,
            parent_message_uuid=self.parent_message_id,
            tool_name=self.tool_name,
            tool_uuid=self.tool_uuid,
            tool_type=self.tool_type,
            oai_tool_call_id=self.oai_tool_call_id,
            tool_json_schema=self.tool_json_schema,
            tool_call=self.tool_call,
            usage=usage_entity,
            from_storage=True
        )
    
    @classmethod
    def from_entity(cls, entity: ChatMessage) -> 'ChatMessageSQL':
        """Convert from Entity to SQL model."""
        # Convert UUID objects to strings for JSON serialization
        str_old_ids = [str(uid) for uid in entity.old_ids] if entity.old_ids else []
        
        return cls(
            ecs_id=entity.ecs_id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            created_at=entity.created_at,
            old_ids=str_old_ids,  # Use string representation for JSON serialization
            timestamp=entity.timestamp,
            role=entity.role.value,  # Convert enum to string
            content=entity.content,
            author_uuid=entity.author_uuid,
            chat_thread_id=entity.chat_thread_id,
            parent_message_id=entity.parent_message_uuid,
            tool_name=entity.tool_name,
            tool_uuid=entity.tool_uuid,
            tool_type=entity.tool_type,
            oai_tool_call_id=entity.oai_tool_call_id,
            tool_json_schema=entity.tool_json_schema,
            tool_call=entity.tool_call,
            entity_type="chat_message"  # Required field
        )
        
    def handle_relationships(self, entity: ChatMessage, session: Session, orm_objects: Dict[UUID, Any]) -> None:
        """Handle relationships for ChatMessage."""
        # Handle chat_thread relationship
        if entity.chat_thread_id:
            if entity.chat_thread_id in orm_objects:
                self.chat_thread = orm_objects[entity.chat_thread_id]
            else:
                # Try to find in database
                chat_thread = session.query(ChatThreadSQL).filter(
                    ChatThreadSQL.ecs_id == entity.chat_thread_id
                ).first()
                if chat_thread:
                    self.chat_thread = chat_thread
        
        # Handle parent_message relationship
        if entity.parent_message_uuid:
            if entity.parent_message_uuid in orm_objects:
                self.parent_message = orm_objects[entity.parent_message_uuid]
            else:
                # Try to find in database
                parent_message = session.query(ChatMessageSQL).filter(
                    ChatMessageSQL.ecs_id == entity.parent_message_uuid
                ).first()
                if parent_message:
                    self.parent_message = parent_message
        
        # Handle tool relationship by UUID
        if entity.tool_uuid:
            if entity.tool_uuid in orm_objects:
                self.tool = orm_objects[entity.tool_uuid]
            else:
                # Try to find in database
                tool = session.query(ToolSQL).filter(
                    ToolSQL.ecs_id == entity.tool_uuid
                ).first()
                if tool:
                    self.tool = tool
        
        # Handle usage relationship
        if entity.usage:
            if entity.usage.ecs_id in orm_objects:
                self.usage = orm_objects[entity.usage.ecs_id]
            else:
                # Try to find in database
                usage = session.query(UsageSQL).filter(
                    UsageSQL.ecs_id == entity.usage.ecs_id
                ).first()
                if usage:
                    self.usage = usage

class SystemPromptSQL(EntityBase):
    """SQLAlchemy model for SystemPrompt entities."""
    __tablename__ = "system_prompt"
    
    # SystemPrompt specific fields
    content = mapped_column(Text, nullable=False)
    prompt_name = mapped_column(String(255), nullable=True)  # Maps to 'name' in Entity
    prompt_description = mapped_column(Text, nullable=True)  # Not in current Entity model
    
    __mapper_args__ = {
        "polymorphic_identity": "system_prompt",
    }
    
    def to_entity(self) -> SystemPrompt:
        """Convert from SQL model to Entity."""
        return SystemPrompt(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=self.old_ids,
            name=self.prompt_name,
            content=self.content,
            from_storage=True
        )
    
    @classmethod
    def from_entity(cls, entity: SystemPrompt) -> 'SystemPromptSQL':
        """Convert from Entity to SQL model."""
        # Convert UUID objects to strings for JSON serialization
        str_old_ids = [str(uid) for uid in entity.old_ids] if entity.old_ids else []
        
        return cls(
            ecs_id=entity.ecs_id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            created_at=entity.created_at,
            old_ids=str_old_ids,  # Use string representation for JSON serialization
            prompt_name=entity.name,
            content=entity.content,
            entity_type="system_prompt"  # Required field
        )

class LLMConfigSQL(EntityBase):
    """SQLAlchemy model for LLMConfig entities."""
    __tablename__ = "llm_config"
    
    # LLMConfig specific fields
    model = mapped_column(String(100), nullable=False)
    provider_name = mapped_column(String(50), nullable=True)  # Maps to 'client' in Entity
    provider_api_key = mapped_column(String(255), nullable=True)  # Not a current field
    max_tokens = mapped_column(Integer, nullable=True)
    temperature = mapped_column(Float, nullable=True, default=0.0)  # Using proper Float type
    response_format = mapped_column(JSON, nullable=True)
    llm_config = mapped_column(JSON, nullable=True)  # Stores additional config
    
    # Added missing fields
    use_cache = mapped_column(Boolean, nullable=False, default=True)
    reasoner = mapped_column(Boolean, nullable=False, default=False)
    reasoning_effort = mapped_column(String(20), nullable=False, default="medium")
    
    __mapper_args__ = {
        "polymorphic_identity": "llm_config",
    }
    
    def to_entity(self) -> LLMConfig:
        """Convert from SQL model to Entity."""
        # Handle response format conversion
        response_format_value = None
        if isinstance(self.response_format, str):
            response_format_value = ResponseFormat(self.response_format)
        elif isinstance(self.response_format, dict) and 'value' in self.response_format:
            response_format_value = ResponseFormat(self.response_format['value'])
            
        return LLMConfig(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=self.old_ids,
            client=LLMClient(self.provider_name) if self.provider_name else LLMClient.openai,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format=response_format_value or ResponseFormat.text,
            use_cache=self.use_cache,
            reasoner=self.reasoner,
            reasoning_effort=self.reasoning_effort,
            from_storage=True
        )
    
    @classmethod
    def from_entity(cls, entity: LLMConfig) -> 'LLMConfigSQL':
        """Convert from Entity to SQL model."""
        # Convert UUID objects to strings for JSON serialization
        str_old_ids = [str(uid) for uid in entity.old_ids] if entity.old_ids else []
        
        return cls(
            ecs_id=entity.ecs_id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            created_at=entity.created_at,
            old_ids=str_old_ids,  # Use string representation for JSON serialization
            model=entity.model,
            provider_name=entity.client.value,  # From 'client' in Entity (enum)
            max_tokens=entity.max_tokens,
            temperature=entity.temperature,
            response_format=entity.response_format.value,  # Store enum value
            use_cache=entity.use_cache,
            reasoner=entity.reasoner,
            reasoning_effort=entity.reasoning_effort,
            entity_type="llm_config"  # Required field
        )

class ToolSQL(EntityBase):
    """Base SQLAlchemy model for Tool entities with polymorphic inheritance."""
    __tablename__ = "tool"
    
    # Tool specific fields
    name = mapped_column(String(255), nullable=False)
    tool_description = mapped_column(Text, nullable=True)  # Maps to 'description' in Entity
    tool_parameters_schema = mapped_column(JSON, nullable=True)  # Maps to 'input_schema' in Entity
    
    # Polymorphic discriminator
    tool_type = mapped_column(String(50), nullable=False)
    
    __mapper_args__ = {
        "polymorphic_identity": "tool",
        "polymorphic_on": tool_type
    }
    
    # This will be overridden by subclasses
    def to_entity(self) -> Union[CallableTool, StructuredTool]:
        """Base implementation for tool conversion."""
        # Default to CallableTool if not specialized
        return CallableTool(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=self.old_ids,
            name=self.name,
            docstring=self.tool_description,
            input_schema=self.tool_parameters_schema or {},
            from_storage=True
        )
        
    @classmethod
    def from_entity(cls, entity: Union[CallableTool, StructuredTool]) -> 'ToolSQL':
        """Basic conversion from Entity to SQL model."""
        # Convert UUID objects to strings for JSON serialization
        str_old_ids = [str(uid) for uid in entity.old_ids] if entity.old_ids else []
        
        # This will be called by more specific subclasses
        return cls(
            ecs_id=entity.ecs_id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            created_at=entity.created_at,
            old_ids=str_old_ids,  # Use string representation for JSON serialization
            name=entity.name,
            tool_description=getattr(entity, 'description', None),
            tool_parameters_schema=getattr(entity, 'parameters_schema', {}),
            entity_type="tool",
            tool_type="tool"  # Will be overridden by subclasses
        )

class CallableToolSQL(ToolSQL):
    """SQLAlchemy model for CallableTool entities."""
    __tablename__ = "callable_tool"
    
    # Primary key that links to the parent
    id = mapped_column(Integer, ForeignKey("tool.id"), primary_key=True)
    
    # CallableTool specific fields
    input_schema = mapped_column(JSON, nullable=True)
    output_schema = mapped_column(JSON, nullable=True)
    strict_schema = mapped_column(Boolean, nullable=False, default=True)
    callable_text = mapped_column(Text, nullable=True)
    
    __mapper_args__ = {
        "polymorphic_identity": "callable_tool",
    }
    
    def to_entity(self) -> CallableTool:
        """Convert from SQL model to Entity."""
        return CallableTool(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=self.old_ids,
            name=self.name,
            docstring=self.tool_description,
            input_schema=self.input_schema or {},
            output_schema=self.output_schema or {},
            strict_schema=self.strict_schema,
            callable_text=self.callable_text,
            from_storage=True
        )
    
    @classmethod
    def from_entity(cls, entity: CallableTool) -> 'CallableToolSQL':
        """Convert from Entity to SQL model."""
        # Convert UUID objects to strings for JSON serialization
        str_old_ids = [str(uid) for uid in entity.old_ids] if entity.old_ids else []
        
        return cls(
            ecs_id=entity.ecs_id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            created_at=entity.created_at,
            old_ids=str_old_ids,  # Use string representation for JSON serialization
            name=entity.name,
            tool_description=entity.docstring,
            tool_parameters_schema={},  # Not used directly
            input_schema=entity.input_schema,
            output_schema=entity.output_schema,
            strict_schema=entity.strict_schema,
            callable_text=entity.callable_text,
            entity_type="tool",
            tool_type="callable_tool"
        )

class StructuredToolSQL(ToolSQL):
    """SQLAlchemy model for StructuredTool entities."""
    __tablename__ = "structured_tool"
    
    # Primary key that links to the parent
    id = mapped_column(Integer, ForeignKey("tool.id"), primary_key=True)
    
    # StructuredTool specific fields
    tool_output_schema = mapped_column(JSON, nullable=True)  # Maps to json_schema in Entity
    instruction_string = mapped_column(Text, nullable=True)
    strict_schema = mapped_column(Boolean, nullable=False, default=True)
    
    __mapper_args__ = {
        "polymorphic_identity": "structured_tool",
    }
    
    def to_entity(self) -> StructuredTool:
        """Convert from SQL model to Entity."""
        return StructuredTool(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=self.old_ids,
            name=self.name,
            description=self.tool_description,
            instruction_string=self.instruction_string or "Please follow this JSON schema for your response:",
            json_schema=self.tool_output_schema or {},
            strict_schema=self.strict_schema,
            from_storage=True
        )
    
    @classmethod
    def from_entity(cls, entity: StructuredTool) -> 'StructuredToolSQL':
        """Convert from Entity to SQL model."""
        # Convert UUID objects to strings for JSON serialization
        str_old_ids = [str(uid) for uid in entity.old_ids] if entity.old_ids else []
        
        return cls(
            ecs_id=entity.ecs_id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            created_at=entity.created_at,
            old_ids=str_old_ids,  # Use string representation for JSON serialization
            name=entity.name,
            tool_description=entity.description,
            tool_parameters_schema={},  # Not used in StructuredTool
            tool_output_schema=entity.json_schema,
            instruction_string=entity.instruction_string,
            strict_schema=entity.strict_schema,
            entity_type="tool",
            tool_type="structured_tool"
        )

class UsageSQL(EntityBase):
    """SQLAlchemy model for Usage entities."""
    __tablename__ = "usage"
    
    # Usage specific fields
    prompt_tokens = mapped_column(Integer, nullable=True)
    completion_tokens = mapped_column(Integer, nullable=True)
    total_tokens = mapped_column(Integer, nullable=True)
    model = mapped_column(String(100), nullable=True)
    
    # Added missing fields for various token tracking
    cache_creation_input_tokens = mapped_column(Integer, nullable=True)
    cache_read_input_tokens = mapped_column(Integer, nullable=True)
    accepted_prediction_tokens = mapped_column(Integer, nullable=True)
    audio_tokens = mapped_column(Integer, nullable=True)
    reasoning_tokens = mapped_column(Integer, nullable=True)
    rejected_prediction_tokens = mapped_column(Integer, nullable=True)
    cached_tokens = mapped_column(Integer, nullable=True)
    
    __mapper_args__ = {
        "polymorphic_identity": "usage",
    }
    
    def to_entity(self) -> Usage:
        """Convert from SQL model to Entity."""
        return Usage(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=self.old_ids,
            model=self.model or "unknown",
            prompt_tokens=self.prompt_tokens or 0,
            completion_tokens=self.completion_tokens or 0,
            total_tokens=self.total_tokens or 0,
            cache_creation_input_tokens=self.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens,
            accepted_prediction_tokens=self.accepted_prediction_tokens,
            audio_tokens=self.audio_tokens,
            reasoning_tokens=self.reasoning_tokens,
            rejected_prediction_tokens=self.rejected_prediction_tokens,
            cached_tokens=self.cached_tokens,
            from_storage=True
        )
    
    @classmethod
    def from_entity(cls, entity: Usage) -> 'UsageSQL':
        """Convert from Entity to SQL model."""
        # Convert UUID objects to strings for JSON serialization
        str_old_ids = [str(uid) for uid in entity.old_ids] if entity.old_ids else []
        
        return cls(
            ecs_id=entity.ecs_id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            created_at=entity.created_at,
            old_ids=str_old_ids,  # Use string representation for JSON serialization
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
            cached_tokens=entity.cached_tokens,
            entity_type="usage"  # Required field
        )

class GeneratedJsonObjectSQL(EntityBase):
    """SQLAlchemy model for GeneratedJsonObject entities."""
    __tablename__ = "generated_json_object"
    
    # GeneratedJsonObject specific fields
    obj_data = mapped_column(JSON, nullable=False)  # Maps to 'data' in Entity
    obj_schema = mapped_column(JSON, nullable=True)  # Maps to 'schema' in Entity
    obj_name = mapped_column(String(255), nullable=True)  # Maps to 'name' in Entity
    obj_object = mapped_column(JSON, nullable=True)  # Maps to 'object' in Entity
    tool_call_id = mapped_column(String(255), nullable=True)  # Added missing field
    
    __mapper_args__ = {
        "polymorphic_identity": "generated_json_object",
    }
    
    def to_entity(self) -> GeneratedJsonObject:
        """Convert from SQL model to Entity."""
        return GeneratedJsonObject(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=self.old_ids,
            name=self.obj_name or "unnamed",
            object=self.obj_object or {},
            tool_call_id=self.tool_call_id,
            from_storage=True
        )
    
    @classmethod
    def from_entity(cls, entity: GeneratedJsonObject) -> 'GeneratedJsonObjectSQL':
        """Convert from Entity to SQL model."""
        # Convert UUID objects to strings for JSON serialization
        str_old_ids = [str(uid) for uid in entity.old_ids] if entity.old_ids else []
        
        return cls(
            ecs_id=entity.ecs_id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            created_at=entity.created_at,
            old_ids=str_old_ids,  # Use string representation for JSON serialization
            obj_name=entity.name,
            obj_object=entity.object,
            tool_call_id=entity.tool_call_id,
            obj_data={},  # Default value since current entity doesn't have this
            entity_type="generated_json_object"  # Required field
        )

class RawOutputSQL(EntityBase):
    """SQLAlchemy model for RawOutput entities."""
    __tablename__ = "raw_output"
    
    # RawOutput specific fields
    raw_result = mapped_column(JSON, nullable=False)
    completion_kwargs = mapped_column(JSON, nullable=True)
    start_time = mapped_column(Integer, nullable=True)  # Store as integer milliseconds
    end_time = mapped_column(Integer, nullable=True)    # Store as integer milliseconds 
    chat_thread_id = mapped_column(Uuid, nullable=True)
    chat_thread_live_id = mapped_column(Uuid, nullable=True) 
    client_type = mapped_column(String(50), nullable=True)
    
    __mapper_args__ = {
        "polymorphic_identity": "raw_output",
    }
    
    def to_entity(self) -> RawOutput:
        """Convert from SQL model to Entity."""
        client = LLMClient(self.client_type) if self.client_type else LLMClient.openai
        
        return RawOutput(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=self.old_ids,
            raw_result=self.raw_result or {},
            completion_kwargs=self.completion_kwargs or {},
            start_time=self.start_time or 0,
            end_time=self.end_time or 0,
            chat_thread_id=self.chat_thread_id,
            chat_thread_live_id=self.chat_thread_live_id,
            client=client,
            from_storage=True
        )
    
    @classmethod
    def from_entity(cls, entity: RawOutput) -> 'RawOutputSQL':
        """Convert from Entity to SQL model."""
        # Convert UUID objects to strings for JSON serialization
        str_old_ids = [str(uid) for uid in entity.old_ids] if entity.old_ids else []
        
        return cls(
            ecs_id=entity.ecs_id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            created_at=entity.created_at,
            old_ids=str_old_ids,  # Use string representation for JSON serialization
            raw_result=entity.raw_result,
            completion_kwargs=entity.completion_kwargs,
            start_time=entity.start_time,
            end_time=entity.end_time,
            chat_thread_id=entity.chat_thread_id,
            chat_thread_live_id=entity.chat_thread_live_id,
            client_type=entity.client.value if entity.client else None,
            entity_type="raw_output"  # Required field
        )

class ProcessedOutputSQL(EntityBase):
    """SQLAlchemy model for ProcessedOutput entities."""
    __tablename__ = "processed_output"
    
    # Relationships
    raw_output_id = mapped_column(Uuid, ForeignKey("raw_output.ecs_id"), nullable=True)
    raw_output = relationship("RawOutputSQL")
    json_object_id = mapped_column(Uuid, ForeignKey("generated_json_object.ecs_id"), nullable=True)
    json_object = relationship("GeneratedJsonObjectSQL") 
    usage_id = mapped_column(Uuid, ForeignKey("usage.ecs_id"), nullable=True)
    usage = relationship("UsageSQL")
    
    # ProcessedOutput specific fields
    content = mapped_column(Text, nullable=True)
    error = mapped_column(Text, nullable=True)
    time_taken = mapped_column(Integer, nullable=True)  # Store as integer milliseconds
    llm_client_type = mapped_column(String(50), nullable=True)
    chat_thread_id = mapped_column(Uuid, nullable=True)
    chat_thread_live_id = mapped_column(Uuid, nullable=True)
    
    __mapper_args__ = {
        "polymorphic_identity": "processed_output",
    }
    
    def to_entity(self) -> ProcessedOutput:
        """Convert from SQL model to Entity."""
        # Load related entities if available
        raw_output = self.raw_output.to_entity() if self.raw_output else None
        json_object = self.json_object.to_entity() if self.json_object else None
        usage = self.usage.to_entity() if self.usage else None
        
        # Handle LLM client type conversion
        client = LLMClient(self.llm_client_type) if self.llm_client_type else LLMClient.openai
        
        # Required field validations
        if raw_output is None:
            # Create a minimal RawOutput if missing
            raw_output = RawOutput(
                raw_result={},
                completion_kwargs={},
                start_time=0,
                end_time=0,
                client=client,
                chat_thread_id=self.chat_thread_id,
                chat_thread_live_id=self.chat_thread_live_id
            )
            
        if self.chat_thread_id is None or self.chat_thread_live_id is None:
            raise ValueError("Chat thread ID and live ID are required for ProcessedOutput")
        
        return ProcessedOutput(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=self.old_ids,
            content=self.content,
            json_object=json_object,
            usage=usage,
            error=self.error,
            time_taken=self.time_taken or 0,
            llm_client=client,
            raw_output=raw_output,
            chat_thread_id=self.chat_thread_id,
            chat_thread_live_id=self.chat_thread_live_id,
            from_storage=True
        )
    
    @classmethod
    def from_entity(cls, entity: ProcessedOutput) -> 'ProcessedOutputSQL':
        """Convert from Entity to SQL model."""
        # Convert UUID objects to strings for JSON serialization
        str_old_ids = [str(uid) for uid in entity.old_ids] if entity.old_ids else []
        
        return cls(
            ecs_id=entity.ecs_id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            created_at=entity.created_at,
            old_ids=str_old_ids,  # Use string representation for JSON serialization
            content=entity.content,
            error=entity.error,
            time_taken=entity.time_taken,
            llm_client_type=entity.llm_client.value if entity.llm_client else None,
            chat_thread_id=entity.chat_thread_id,
            chat_thread_live_id=entity.chat_thread_live_id,
            entity_type="processed_output"  # Required field
        )
        
    def handle_relationships(self, entity: ProcessedOutput, session: Session, orm_objects: Dict[UUID, Any]) -> None:
        """Handle relationships for ProcessedOutput."""
        # Handle raw_output relationship
        if entity.raw_output:
            if entity.raw_output.ecs_id in orm_objects:
                self.raw_output = orm_objects[entity.raw_output.ecs_id]
            else:
                # Try to find in database
                raw_output = session.query(RawOutputSQL).filter(
                    RawOutputSQL.ecs_id == entity.raw_output.ecs_id
                ).first()
                if raw_output:
                    self.raw_output = raw_output
                    
        # Handle json_object relationship
        if entity.json_object:
            if entity.json_object.ecs_id in orm_objects:
                self.json_object = orm_objects[entity.json_object.ecs_id]
            else:
                # Try to find in database
                json_object = session.query(GeneratedJsonObjectSQL).filter(
                    GeneratedJsonObjectSQL.ecs_id == entity.json_object.ecs_id
                ).first()
                if json_object:
                    self.json_object = json_object
                    
        # Handle usage relationship
        if entity.usage:
            if entity.usage.ecs_id in orm_objects:
                self.usage = orm_objects[entity.usage.ecs_id]
            else:
                # Try to find in database
                usage = session.query(UsageSQL).filter(
                    UsageSQL.ecs_id == entity.usage.ecs_id
                ).first()
                if usage:
                    self.usage = usage

class RequestLimitsSQL(EntityBase):
    """SQLAlchemy model for RequestLimits entities."""
    __tablename__ = "request_limits"
    
    # RequestLimits specific fields
    max_requests_per_minute = mapped_column(Integer, nullable=False, default=50)
    max_tokens_per_minute = mapped_column(Integer, nullable=False, default=100000)
    provider = mapped_column(String(20), nullable=False, default="openai")
    
    __mapper_args__ = {
        "polymorphic_identity": "request_limits",
    }
    
    @classmethod
    def from_entity(cls, entity: RequestLimits) -> 'RequestLimitsSQL':
        """Convert from Entity to SQL model."""
        return cls(
            ecs_id=entity.ecs_id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            created_at=entity.created_at,
            old_ids=[str(uid) for uid in entity.old_ids] if entity.old_ids else [],
            max_requests_per_minute=entity.max_requests_per_minute,
            max_tokens_per_minute=entity.max_tokens_per_minute,
            provider=entity.provider,
            entity_type="request_limits"
        )
    
    def to_entity(self) -> RequestLimits:
        """Convert from SQL model to Entity."""
        return RequestLimits(
            ecs_id=self.ecs_id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            old_ids=[UUID(uid) for uid in self.old_ids] if self.old_ids else [],
            max_requests_per_minute=self.max_requests_per_minute,
            max_tokens_per_minute=self.max_tokens_per_minute,
            provider=self.provider,
            from_storage=True
        )

# Class to ORM model mapping
ENTITY_MODEL_MAP = {
    ChatThread: ChatThreadSQL,
    ChatMessage: ChatMessageSQL,
    SystemPrompt: SystemPromptSQL,
    LLMConfig: LLMConfigSQL,
    CallableTool: CallableToolSQL,
    StructuredTool: StructuredToolSQL,
    Usage: UsageSQL,
    GeneratedJsonObject: GeneratedJsonObjectSQL,
    RawOutput: RawOutputSQL,
    ProcessedOutput: ProcessedOutputSQL,
    RequestLimits: RequestLimitsSQL,
    Entity: BaseEntitySQL
}