
# Versioned Entity ORM with SQLAlchemy

This guide introduces our versioned entity ORM system built on SQLAlchemy. Unlike traditional ORM systems that maintain a 1:1 mapping between objects and database rows, our system implements a versioning approach where entities can evolve through multiple versions while maintaining lineage relationships.

## Key Concepts

- **Dual Identity**: Each entity has both a database `id` (integer primary key) and a `ecs_id` (UUID) for version tracking
- **Entity Lineage**: Related versions of the same logical entity are connected via `lineage_id` and `parent_id`
- **Entity-to-ORM Conversion**: Domain entities are converted to/from SQL models via `to_entity()` and `from_entity()`
- **Hierarchical Relationships**: The system efficiently handles complex nested entity structures
- **Automatic Versioning**: Changes to entities trigger version creation with proper lineage tracking

## Versioning and Relationships

Our system handles relationships differently than standard ORM systems:

1. **Relationship Version Tracking**: When related entities change, the relationship is maintained through versioning
2. **Bottom-Up Propagation**: Changes to child entities can trigger parent entity versioning
3. **ID-Based References**: Relationships reference UUIDs, not database integer IDs
4. **Two-Phase Loading**: Relationships are loaded and processed in two phases to maintain integrity

## Declare ORM Models

The first step is defining our base classes and ORM models:

```python
import logging
from typing import Dict, Any, Optional, List, Union, Type, Set
from uuid import UUID, uuid4
from datetime import datetime, UTC
import json

from sqlalchemy import (
    Column, Integer, String, Boolean, Float, ForeignKey, Table, 
    JSON, Text, DateTime, create_engine, select
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session, sessionmaker, joinedload

# Base class for all SQL models
Base = declarative_base()

# Convert UUIDs to strings for JSON storage
def convert_uuids_to_strings(data: Any) -> Any:
    """Recursively convert all UUIDs to strings in any data structure."""
    if isinstance(data, UUID):
        return str(data)
    elif isinstance(data, dict):
        return {k: convert_uuids_to_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_uuids_to_strings(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_uuids_to_strings(item) for item in item)
    else:
        return data

class EntityBase(Base):
    """Abstract base class for all entity-backed models"""
    __abstract__ = True
    
    # Database primary key (integer, auto-incremented)
    id = Column(Integer, primary_key=True)
    
    # Entity versioning fields
    ecs_id = Column(String(36), index=True, nullable=False)
    lineage_id = Column(String(36), index=True, nullable=False)
    parent_id = Column(String(36), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    old_ids = Column(JSON, default=list)
    
    def to_entity(self):
        """Convert SQL model to domain entity. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement to_entity()")
    
    @classmethod
    def from_entity(cls, entity):
        """Create SQL model from domain entity. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement from_entity()")
    
    def handle_relationships(self, entity, session, orm_objects=None):
        """Handle relationships. May be implemented by subclasses that need it."""
        pass
```

## Define Domain Entity Models

Now we'll create a simplified example of a versioned Chat Thread model:

```python
# Many-to-many link between ChatThread and ChatMessage
thread_message_link = Table(
    'thread_message_link', 
    Base.metadata,
    Column('thread_id', Integer, ForeignKey('chat_thread.id'), primary_key=True),
    Column('message_id', Integer, ForeignKey('chat_message.id'), primary_key=True)
)

class ChatMessage(EntityBase):
    """SQL model for a single ChatMessage entity"""
    __tablename__ = 'chat_message'
    
    # Message content
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC))
    
    # Message threading (different from versioning)
    parent_message_uuid = Column(String(36), nullable=True)
    
    # Relationships
    threads = relationship(
        "ChatThread",
        secondary=thread_message_link,
        back_populates="messages"
    )
    
    def to_entity(self):
        """Convert to domain entity"""
        from minference.threads.models import ChatMessage, MessageRole
        
        return ChatMessage(
            ecs_id=UUID(self.ecs_id),
            lineage_id=UUID(self.lineage_id),
            parent_id=UUID(self.parent_id) if self.parent_id else None,
            created_at=self.created_at,
            old_ids=[UUID(id_str) for id_str in self.old_ids],
            role=MessageRole(self.role),
            content=self.content,
            timestamp=self.timestamp,
            parent_message_uuid=UUID(self.parent_message_uuid) if self.parent_message_uuid else None,
            from_storage=True
        )

    @classmethod
    def from_entity(cls, entity):
        """Create from domain entity"""
        from minference.threads.models import ChatMessage
        
        return cls(
            ecs_id=str(entity.ecs_id),
            lineage_id=str(entity.lineage_id),
            parent_id=str(entity.parent_id) if entity.parent_id else None,
            created_at=entity.created_at,
            old_ids=[str(id) for id in entity.old_ids],
            role=entity.role.value,
            content=entity.content,
            timestamp=entity.timestamp,
            parent_message_uuid=str(entity.parent_message_uuid) if entity.parent_message_uuid else None
        )

class ChatThread(EntityBase):
    """SQL model for the ChatThread entity"""
    __tablename__ = 'chat_thread'
    
    # Domain fields
    name = Column(String(255))
    use_history = Column(Boolean, default=True)
    
    # Relationships
    messages = relationship(
        "ChatMessage",
        secondary=thread_message_link,
        back_populates="threads",
        order_by="ChatMessage.timestamp"
    )

    def to_entity(self):
        """Convert to domain entity"""
        from minference.threads.models import ChatThread
        
        # Convert message relationships
        message_entities = [m.to_entity() for m in self.messages]
        
        return ChatThread(
            ecs_id=UUID(self.ecs_id),
            lineage_id=UUID(self.lineage_id),
            parent_id=UUID(self.parent_id) if self.parent_id else None,
            created_at=self.created_at,
            old_ids=[UUID(id_str) for id_str in self.old_ids],
            name=self.name,
            history=message_entities,
            use_history=self.use_history,
            from_storage=True
        )

    @classmethod
    def from_entity(cls, entity):
        """Create from domain entity"""
        from minference.threads.models import ChatThread
        
        # Create the basic thread instance without relationships
        return cls(
            ecs_id=str(entity.ecs_id),
            lineage_id=str(entity.lineage_id),
            parent_id=str(entity.parent_id) if entity.parent_id else None,
            created_at=entity.created_at,
            old_ids=[str(id) for id in entity.old_ids],
            name=entity.name,
            use_history=entity.use_history
        )
    
    def handle_relationships(self, entity, session, orm_objects=None):
        """Handle relationships between entities"""
        orm_objects = orm_objects or {}
        
        # Handle messages relationship
        self.messages = []
        for message in entity.history:
            message_id = str(message.ecs_id)
            if message_id in orm_objects:
                self.messages.append(orm_objects[message_id])
```

## Entity-ORM Mapping Registry

We need to maintain a mapping between entity types and their ORM models:

```python
# Import domain entities
from minference.threads.models import ChatThread as ChatThreadEntity
from minference.threads.models import ChatMessage as ChatMessageEntity

# Create entity to ORM mapping
ENTITY_ORM_MAP = {
    ChatThreadEntity: ChatThread,
    ChatMessageEntity: ChatMessage,
    # Add other entity mappings here
}
```

## Create SQL Storage Engine

Now we'll set up the SQLAlchemy engine and configure our storage:

```python
def create_database_and_session_factory(connection_string):
    """Create database engine and session factory"""
    engine = create_engine(connection_string, echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

def setup_sql_storage(connection_string):
    """Set up SQL storage for EntityRegistry"""
    from minference.ecs.entity import EntityRegistry
    from minference.sql_models import ENTITY_ORM_MAP
    
    session_factory = create_database_and_session_factory(connection_string)
    sql_storage = SqlEntityStorage(session_factory, ENTITY_ORM_MAP)
    EntityRegistry.use_storage(sql_storage)
    return sql_storage
```

## Working with Versioned Entities

Let's see how to use our versioned entity system:

### Creating and Storing Entities

```python
# Set up the storage engine
storage = setup_sql_storage("sqlite:///test.db")

# Create a new chat thread
thread = ChatThreadEntity(name="Test Thread")

# Create and add messages
message1 = ChatMessageEntity(
    role=MessageRole.user,
    content="Hello, world!",
    chat_thread_id=thread.ecs_id
)

message2 = ChatMessageEntity(
    role=MessageRole.assistant,
    content="Hi there! How can I help?",
    chat_thread_id=thread.ecs_id,
    parent_message_uuid=message1.ecs_id
)

# Add messages to thread
thread.history.append(message1)
thread.history.append(message2)

# The thread and its messages are automatically registered with EntityRegistry
# due to Pydantic's model_validator
```

### Querying for Entities

```python
# Get a thread by its UUID
thread_id = thread.ecs_id
retrieved_thread = ChatThreadEntity.get(thread_id)

# Access the related messages
for message in retrieved_thread.history:
    print(f"{message.role}: {message.content}")
```

### Making Changes and Versioning

```python
# Modify a message (will create a new version)
message = retrieved_thread.history[1]
message.content = "Updated response"

# Fork the message to create a new version
new_message = message.fork()

# The old version is preserved in history
# Both are linked by the same lineage_id but have different ecs_id values
print(f"Original message ID: {message.ecs_id}")
print(f"New message ID: {new_message.ecs_id}")
print(f"Both have lineage ID: {message.lineage_id}")

# Get all versions in the lineage
lineage = EntityRegistry.get_lineage_entities(message.lineage_id)
for version in lineage:
    print(f"Version {version.ecs_id}: {version.content}")
```

## Best Practices for Versioned Entity ORM

### 1. UUID Handling

Always convert UUIDs to strings when storing in database:

```python
# In from_entity method
return cls(
    ecs_id=str(entity.ecs_id),
    lineage_id=str(entity.lineage_id),
    # ...other fields
)

# In to_entity method
return EntityClass(
    ecs_id=UUID(self.ecs_id),
    lineage_id=UUID(self.lineage_id),
    # ...other fields
)
```

### 2. Relationship Handling

Use the two-phase approach for handling relationships:

```python
# Phase 1: Create all entity objects first
orm_objects = {}
for entity_id, entity in entities_to_store.items():
    orm_obj = orm_cls.from_entity(entity)
    session.add(orm_obj)
    orm_objects[entity_id] = orm_obj

session.flush()  # Assign database IDs

# Phase 2: Set up relationships
for entity_id, orm_obj in orm_objects.items():
    entity = entities_to_store[entity_id]
    orm_obj.handle_relationships(entity, session, orm_objects)
```

### 3. Eager Loading

Always use eager loading to avoid N+1 query problems:

```python
# When querying chat threads, load messages eagerly
query = session.query(ChatThread).options(
    joinedload(ChatThread.messages)
)
```

### 4. Error Handling and Debugging

Include helpful error messages and debugging in ORM code:

```python
def handle_relationships(self, entity, session, orm_objects=None):
    """Handle relationships with detailed error messages"""
    try:
        orm_objects = orm_objects or {}
        # Relationship handling logic here
    except Exception as e:
        logger.error(f"Error handling relationships for {entity.__class__.__name__}({entity.ecs_id}): {str(e)}")
        logger.debug("Full traceback:", exc_info=True)
        raise
```

### 5. Type Validation

Always validate entity types in from_entity methods:

```python
@classmethod
def from_entity(cls, entity):
    from minference.threads.models import ChatThread
    
    if not isinstance(entity, ChatThread):
        raise TypeError(f"Expected ChatThread, got {type(entity).__name__}")
    
    # Continue with conversion
```

## Advanced Topics

### Querying Entity Lineage

```python
# Get all versions of an entity
lineage_id = entity.lineage_id
versions = EntityRegistry.get_lineage_entities(lineage_id)

# Sort by creation time
sorted_versions = sorted(versions, key=lambda e: e.created_at)

# Find the latest version
latest_version = sorted_versions[-1]
```

### Handling Complex Hierarchies

For deep entity hierarchies, use recursive collection:

```python
def collect_entities(entity, collected=None):
    """Recursively collect all entities in a hierarchy"""
    if collected is None:
        collected = {}
    
    entity_id = str(entity.ecs_id)
    if entity_id not in collected:
        collected[entity_id] = entity
        
        # Process all nested entities
        for sub_entity in entity.get_sub_entities():
            collect_entities(sub_entity, collected)
            
    return collected

# Use the collector
all_entities = collect_entities(root_entity)
```

### Polymorphic Entity Storage

For entities that share a table but have different behaviors, SQLAlchemy's single-table inheritance is ideal:

```python
class Tool(EntityBase):
    """Base class for different tool types"""
    __tablename__ = 'tool'
    
    # Discriminator column for type
    tool_type = Column(String(20), nullable=False)
    
    # Common fields for all tools
    name = Column(String(255), nullable=False)
    strict_schema = Column(Boolean, default=True)
    
    # Define polymorphic identity mechanism
    __mapper_args__ = {
        'polymorphic_on': tool_type,
        'polymorphic_identity': 'base'
    }
    
    # Relationships
    threads = relationship(
        "ChatThread",
        secondary="thread_tool_link",
        back_populates="tools", 
        lazy="joined"
    )
    
    def to_entity(self):
        """
        Convert to appropriate domain entity based on polymorphic type
        This will be overridden by subclasses
        """
        raise NotImplementedError(
            "Tool is an abstract base class, use a specific subclass"
        )


class CallableTool(Tool):
    """For callable function tools"""
    # CallableTool-specific columns
    docstring = Column(Text)
    callable_text = Column(Text)
    input_schema = Column(JSON, default=dict)
    output_schema = Column(JSON, default=dict)
    
    __mapper_args__ = {
        'polymorphic_identity': 'callable'
    }
    
    def to_entity(self):
        """Convert to CallableTool domain entity"""
        from minference.threads.models import CallableTool
        
        return CallableTool(
            ecs_id=UUID(self.ecs_id),
            lineage_id=UUID(self.lineage_id),
            parent_id=UUID(self.parent_id) if self.parent_id else None,
            created_at=self.created_at,
            old_ids=[UUID(id_str) for id_str in self.old_ids],
            name=self.name,
            strict_schema=self.strict_schema,
            docstring=self.docstring,
            callable_text=self.callable_text,
            input_schema=self.input_schema or {},
            output_schema=self.output_schema or {},
            from_storage=True
        )


class StructuredTool(Tool):
    """For schema-based tools"""
    # StructuredTool-specific columns
    json_schema = Column(JSON)
    description = Column(Text)
    instruction_string = Column(Text)
    
    __mapper_args__ = {
        'polymorphic_identity': 'structured'
    }
    
    def to_entity(self):
        """Convert to StructuredTool domain entity"""
        from minference.threads.models import StructuredTool
        
        return StructuredTool(
            ecs_id=UUID(self.ecs_id),
            lineage_id=UUID(self.lineage_id),
            parent_id=UUID(self.parent_id) if self.parent_id else None,
            created_at=self.created_at,
            old_ids=[UUID(id_str) for id_str in self.old_ids],
            name=self.name,
            strict_schema=self.strict_schema,
            description=self.description or "",
            instruction_string=self.instruction_string or "Please follow this JSON schema for your response:",
            json_schema=self.json_schema or {},
            from_storage=True
        )
```

## Association Object Pattern

For many-to-many relationships with extra data, use the association object pattern:

```python
class MessageUsage(EntityBase):
    """Association object between ChatMessage and Usage"""
    __tablename__ = 'message_usage'
    
    # Primary key columns referencing both sides
    message_id = Column(Integer, ForeignKey('chat_message.id'), primary_key=True)
    usage_id = Column(Integer, ForeignKey('usage.id'), primary_key=True)
    
    # Extra data for the association
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC))
    context_size = Column(Integer)
    
    # Relationships to both sides
    message = relationship("ChatMessage", back_populates="usage_associations")
    usage = relationship("Usage", back_populates="message_associations")
    
    def to_entity(self):
        """Convert to domain entity"""
        from minference.threads.models import MessageUsage
        
        return MessageUsage(
            ecs_id=UUID(self.ecs_id),
            lineage_id=UUID(self.lineage_id),
            parent_id=UUID(self.parent_id) if self.parent_id else None,
            created_at=self.created_at,
            old_ids=[UUID(id_str) for id_str in self.old_ids],
            message_id=UUID(self.message.ecs_id) if self.message else None,
            usage_id=UUID(self.usage.ecs_id) if self.usage else None,
            timestamp=self.timestamp,
            context_size=self.context_size,
            from_storage=True
        )
```

## Using Late-Evaluated Relationships

For circular dependencies or complex relationship configurations, use late evaluation:

```python
class Parent(EntityBase):
    __tablename__ = 'parent_table'
    
    # Late-evaluated relationship using lambda
    children = relationship(
        lambda: Child,
        primaryjoin=lambda: Parent.id == Child.parent_id,
        back_populates="parent"
    )
    
    # String-based relationship definition (evaluated with eval)
    # Only use with trusted code - NEVER with user input
    siblings = relationship(
        "Parent",
        secondary="parent_association",
        primaryjoin="Parent.id == parent_association.c.left_id",
        secondaryjoin="Parent.id == parent_association.c.right_id"
    )
```

## Working with Versioned Entity Lineage

Our versioned entity system adds a layer of complexity but offers powerful tracking:

```python
def get_entity_history(entity_id, session):
    """Get the full history/lineage of an entity"""
    # First get the entity to find its lineage_id
    entity = session.query(EntityBase).filter_by(ecs_id=str(entity_id)).first()
    if not entity:
        return []
        
    # Get all versions with the same lineage_id
    lineage = session.query(type(entity)).filter_by(
        lineage_id=entity.lineage_id
    ).order_by(
        # Order by created_at to get chronological order
        EntityBase.created_at
    ).all()
    
    return lineage
    
def get_entity_children_across_versions(entity_id, session):
    """
    Get children of an entity, including those attached to
    different versions in the same lineage
    """
    # Get all versions of the entity
    entity_versions = get_entity_history(entity_id, session)
    if not entity_versions:
        return []
        
    # Collect children from all versions
    children = []
    for version in entity_versions:
        # Adapt based on your specific relationship structure
        if hasattr(version, 'children'):
            children.extend(version.children)
            
    # Deduplicate by lineage_id to get unique logical entities
    unique_children = {}
    for child in children:
        if child.lineage_id not in unique_children:
            unique_children[child.lineage_id] = child
            
    return list(unique_children.values())
```

This guide introduces the core concepts of our versioned entity ORM system. By following these patterns and best practices, you can efficiently work with complex, versioned entity hierarchies while maintaining the benefits of SQLAlchemy's robust ORM capabilities.