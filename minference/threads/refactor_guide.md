# Versioned Entity System: A Conceptual Guide

## Introduction: Why We Need Entity Versioning

In traditional application development, database records typically have a one-to-one relationship with application objects. When you make a change to an object, you simply update the corresponding database row. This works well for many applications, but falls short when you need to:

- Track the history of changes to entities
- Branch off different versions of the same logical entity
- Maintain complex object hierarchies where changes in child objects affect parents
- Build audit trails that capture not just what changed but the context of those changes

Consider a document editing application. When multiple users collaborate on a document, you might want to track:
- Who made each change
- When changes occurred
- The ability to revert to previous versions
- The option to branch off variations of the document

Our versioned entity system solves these problems by fundamentally rethinking how we map application objects to database records.

## Core Concept #1: Entity Identity vs. Version Identity

In a traditional ORM system:
```
User Object (id=42) ↔ Database Row (id=42)
```

In our versioned entity system:
```
User Entity (lineage_id=A123) → Version 1 (ecs_id=V001)
                              → Version 2 (ecs_id=V002)
                              → Version 3 (ecs_id=V003)
```

Each **logical entity** has a consistent **lineage ID** that remains the same across all versions. Each **version** has its own unique **version ID** (ecs_id).

Think of it like Git:
- A Git repository represents a logical entity (lineage_id)
- Each commit represents a specific version (ecs_id)
- Commits form a history, with each pointing to its parent

When you modify an entity, instead of updating the existing version, you create a new version with a new ecs_id. The new version references its parent version, creating a chain of changes.

## Core Concept #2: Dual Identity System

Our system uses two parallel identification systems:

1. **Database Identity**: Integer primary keys in the database (id)
   - Used for traditional database relationships
   - Optimized for database performance
   - Invisible to application code

2. **Entity Identity**: UUID-based identifiers (ecs_id, lineage_id)
   - Used by application code
   - Preserved across database boundaries
   - Enables versioning and lineage tracking

This dual identity system gives us the performance benefits of integer primary keys while maintaining the versioning capabilities of UUID-based identifiers.

Example:
```
Database Table:
| id | ecs_id | lineage_id | parent_id | data |
|----|--------|------------|-----------|------|
| 1  | abc123 | xyz789     | null      | v1   |
| 2  | def456 | xyz789     | abc123    | v2   |
```

In the application, we'd reference these entities by their ecs_id (abc123, def456), but in the database, relationships would use the integer id (1, 2).

## Core Concept #3: Cold Snapshots and Warm Copies

Our system distinguishes between two states of an entity:

1. **Cold Snapshots**: Immutable versions stored in the database
   - Complete, self-contained records
   - Never modified once created
   - Represent specific points in time

2. **Warm Copies**: Working versions in application memory
   - Can be modified
   - Tracked for changes
   - Fork into new versions when changes are detected

This separation ensures that versioning happens automatically. When you modify a warm copy, the system detects these changes and forks a new version rather than modifying the original.

Think of cold snapshots as photographs (permanent records of a moment) and warm copies as the actual scene (which can change and evolve).

## Core Concept #4: Hierarchical Entity Structures

In real applications, entities often form complex hierarchies. For example, a chat thread might contain:
- Messages
- Attachments
- User references
- Metadata

Our system handles these hierarchies by tracking "sub-entities" - entities that are logically contained within a parent entity. When a sub-entity changes, these changes can propagate up to the parent.

For example, if you edit a message within a chat thread:
1. The message gets a new version
2. The chat thread also gets a new version that references the new message version

This maintains consistency across the entire object graph and ensures that you can always access a consistent snapshot of the entire hierarchy at any point in time.

## From Concept to Implementation: How It Works

Let's walk through how these concepts translate into actual system behavior:

### 1. Entity Creation

When you create a new entity:
1. The system assigns a new lineage_id and ecs_id (both UUIDs)
2. The entity is registered with the EntityRegistry
3. The EntityRegistry creates a cold snapshot and stores it
4. Your application keeps working with the warm copy

```python
# Creating a new entity
thread = ChatThread(name="Discussion")  # new lineage_id and ecs_id assigned
message = ChatMessage(content="Hello")  # new lineage_id and ecs_id assigned

# Adding the message to the thread
thread.history.append(message)

# Both entities are automatically registered with EntityRegistry
```

### 2. Entity Modification

When you modify an entity:
1. Changes are made to the warm copy in memory
2. The system compares the warm copy with the stored cold snapshot
3. If differences are detected, a new version is created (forking)
4. The new version gets a new ecs_id but keeps the same lineage_id
5. The new version stores a reference to its parent version (parent_id)

```python
# Load an entity
thread = ChatThread.get(thread_id)  # Gets a warm copy

# Modify it
thread.name = "Updated Discussion"  

# When you're done, fork to create a new version
new_thread = thread.fork()  # Creates a new version with new ecs_id
```

### 3. Hierarchical Changes

When you modify a sub-entity within a hierarchy:
1. The sub-entity gets a new version
2. All parent entities in the hierarchy also get new versions
3. References are updated to maintain consistency

```python
# Load a thread with messages
thread = ChatThread.get(thread_id)  # Gets a warm copy with its messages

# Modify a message
thread.history[0].content = "Updated message"

# Forking the thread will automatically fork modified sub-entities
new_thread = thread.fork()  # Creates new versions of both thread and message
```

## Advanced Feature: Automatic Entity Tracing

One powerful feature of our system is automatic entity tracing. By using the `@entity_tracer` decorator on functions, the system automatically:

1. Tracks which entities are used by the function
2. Detects modifications to those entities
3. Creates new versions as needed
4. Updates references to maintain consistency

This means you don't have to explicitly call `.fork()` - the system handles versioning automatically when you use traced functions.

```python
@entity_tracer
def update_message(thread_id, message_index, new_content):
    thread = ChatThread.get(thread_id)
    thread.history[message_index].content = new_content
    return thread  # New version created automatically
```

## SQL Implementation: How It Maps to the Database

Our system uses SQLAlchemy to map entities to database tables. Each entity class has a corresponding SQL model that handles:

1. **Conversion**: Translating between entity objects and database records
2. **Relationships**: Managing connections between entities
3. **Versioning**: Tracking lineage and parent-child relationships

The SQL models include:
- Base fields for versioning (ecs_id, lineage_id, parent_id, old_ids)
- Domain-specific fields for the entity
- Relationship configurations
- Conversion methods (to_entity, from_entity)
- Relationship handlers

Example simplified SQL model:

```python
class ChatThreadSQL(EntityBase):
    __tablename__ = 'chat_thread'
    
    # Database primary key
    id = Column(Integer, primary_key=True)
    
    # Entity versioning fields
    ecs_id = Column(String(36), index=True)
    lineage_id = Column(String(36), index=True)
    parent_id = Column(String(36))
    
    # Domain fields
    name = Column(String(255))
    
    # Relationships
    messages = relationship(
        "ChatMessageSQL",
        secondary="thread_message_link",
        lazy="joined"
    )
    
    def to_entity(self):
        """Convert SQL model to domain entity"""
        # Convert database record to entity object
        
    @classmethod
    def from_entity(cls, entity):
        """Create SQL model from domain entity"""
        # Convert entity object to database record
```

## Practical Use Cases

Let's explore some practical examples of how this system enables powerful features:

### Version History and Audit Trails

You can easily retrieve the complete history of an entity:

```python
# Get all versions of a thread
thread_lineage = EntityRegistry.get_lineage_entities(thread.lineage_id)

# Sort by creation time
versions = sorted(thread_lineage, key=lambda t: t.created_at)

# Display version history
for version in versions:
    print(f"Version {version.ecs_id}: {version.name}")
    print(f"  Created: {version.created_at}")
    print(f"  Parent: {version.parent_id}")
```

### Comparison Between Versions

You can compare different versions to see what changed:

```python
# Get two versions
v1 = ChatThread.get(version1_id)
v2 = ChatThread.get(version2_id)

# Compare them
is_different, diffs = v1.compare_entity_fields(v2)

# Show differences
for field, diff in diffs.items():
    print(f"Field '{field}' changed: {diff['old']} → {diff['new']}")
```

### Branching and Merging

Like Git, you can create branches by forking from any version:

```python
# Load a specific version
v1 = ChatThread.get(version1_id)

# Create a branch by modifying it
v1.name = "Branch A"
branch_a = v1.fork()

# Create another branch
v1 = ChatThread.get(version1_id)  # Get a fresh copy
v1.name = "Branch B"
branch_b = v1.fork()
```

## Advanced Concept: Relationship Patterns in a Versioned System

Traditional relationship patterns (one-to-many, many-to-many, etc.) work differently in a versioned system. Let's explore how:

### One-to-Many Relationships

In a one-to-many relationship, the "one" side might reference different versions of the "many" items across different versions:

```
ThreadV1 → [MessageA-V1, MessageB-V1]
ThreadV2 → [MessageA-V1, MessageB-V2, MessageC-V1]
ThreadV3 → [MessageA-V2, MessageB-V2, MessageC-V1]
```

The system tracks these relationships and ensures consistency.

### Many-to-Many Relationships

Many-to-many relationships are handled through association tables, but with version awareness:

```
UserV1 ↔ [ThreadX-V1, ThreadY-V1]
UserV2 ↔ [ThreadX-V1, ThreadY-V2, ThreadZ-V1]
```

This allows entities to participate in multiple relationships while maintaining version integrity.

### Association Objects

For many-to-many relationships with extra data, we use association objects:

```
UserV1 -- UserThreadAssoc-V1(role="admin") --> ThreadX-V1
UserV2 -- UserThreadAssoc-V2(role="viewer") --> ThreadY-V1
```

When either side changes, the association object also gets a new version, preserving the relationship's history.

## Conclusion: The Power of Versioned Entities

Our versioned entity system provides a robust foundation for applications that need:

- **History tracking**: Maintaining complete lineage of changes
- **Audit capabilities**: Knowing who changed what and when
- **Hierarchical integrity**: Ensuring consistency across complex object graphs
- **Branching and comparison**: Creating variations and understanding differences

While more complex than traditional ORM systems, this approach enables sophisticated features that would be difficult or impossible to implement otherwise.

The system handles the complexity of versioning behind the scenes, allowing application code to focus on domain logic rather than version management.


# SQLAlchemy Implementation Guide: Entity Conversion and Relationships

This guide explains how to implement the versioned entity system using SQLAlchemy, focusing on entity conversion and relationship management. We'll explore how domain entities map to database models and how different relationship types are handled in a versioning context.

## 1. Foundation: The SQLAlchemy Interface

Our system uses SQLAlchemy as the bridge between Pydantic domain entities and database tables. The interface consists of:

### 1.1 Base Class Structure

```python
from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref

# The base class for all SQLAlchemy models
Base = declarative_base()

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

### 1.2 Core Interface Requirements

For each domain entity, we create an SQLAlchemy model with:

1. **Conversion Methods**:
   - `to_entity()`: Converts the SQL model to a domain entity
   - `from_entity()`: Creates an SQL model from a domain entity
   - `handle_relationships()`: Manages complex relationships during conversion

2. **Versioning Fields**:
   - `ecs_id`: UUID for the specific version
   - `lineage_id`: UUID for the logical entity across all versions
   - `parent_id`: UUID of the parent version (if any)
   - `old_ids`: List of previous version IDs in the lineage

3. **Relationship Configurations**:
   - Standard SQLAlchemy relationships with versioning awareness
   - Eager loading to prevent N+1 query issues
   - Appropriate cascade settings

## 2. Entity Conversion in Depth

Converting between domain entities and SQL models is more complex in a versioned system. Let's explore how it works:

### 2.1 Basic Domain Entity to SQL Model Conversion

```python
@classmethod
def from_entity(cls, entity):
    """Create an SQL model from a domain entity"""
    # Validate entity type
    if not isinstance(entity, ExpectedEntityType):
        raise TypeError(f"Expected {ExpectedEntityType.__name__}, got {type(entity).__name__}")
    
    # Convert versioning fields
    base_fields = {
        "ecs_id": str(entity.ecs_id),  # Convert UUID to string
        "lineage_id": str(entity.lineage_id),
        "parent_id": str(entity.parent_id) if entity.parent_id else None,
        "created_at": entity.created_at,
        "old_ids": [str(id) for id in entity.old_ids],  # Convert UUID list to string list
    }
    
    # Convert domain-specific fields
    domain_fields = {
        "name": entity.name,
        "description": entity.description,
        # Add other fields specific to this entity
    }
    
    # Create and return the SQL model instance
    return cls(**base_fields, **domain_fields)
```

### 2.2 SQL Model to Domain Entity Conversion

```python
def to_entity(self):
    """Convert SQL model to domain entity"""
    # Import the appropriate entity class
    from your_domain_module import YourEntity
    
    # Convert versioning fields
    base_fields = {
        "ecs_id": UUID(self.ecs_id),  # Convert string to UUID
        "lineage_id": UUID(self.lineage_id),
        "parent_id": UUID(self.parent_id) if self.parent_id else None,
        "created_at": self.created_at,
        "old_ids": [UUID(id_str) for id_str in self.old_ids],  # Convert string list to UUID list
        "from_storage": True  # Indicate this entity came from storage
    }
    
    # Convert domain-specific fields
    domain_fields = {
        "name": self.name,
        "description": self.description,
        # Add other fields specific to this entity
    }
    
    # Create and return the domain entity instance
    return YourEntity(**base_fields, **domain_fields)
```

### 2.3 UUID Handling

Since we use UUIDs for entity identification but store them as strings in the database, proper conversion is essential:

```python
# Helper function for UUID conversion in complex structures
def convert_uuids_to_strings(data):
    """Recursively convert all UUIDs to strings in any data structure."""
    if isinstance(data, UUID):
        return str(data)
    elif isinstance(data, dict):
        return {k: convert_uuids_to_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_uuids_to_strings(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_uuids_to_strings(item) for item in data)
    else:
        return data

# Using it in from_entity
json_data = convert_uuids_to_strings(entity.json_data)
```

### 2.4 JSON Field Handling

For JSON fields that might contain UUIDs or other complex structures:

```python
# In from_entity
sql_model.config_json = convert_uuids_to_strings(entity.config_json)

# In to_entity
if self.config_json:
    # Convert string UUIDs back to UUID objects where needed
    config = self.config_json.copy()
    if 'owner_id' in config:
        config['owner_id'] = UUID(config['owner_id'])
    entity_model.config_json = config
```

## 3. Handling Different Relationship Types

Each relationship type requires special handling in a versioned system:

### 3.1 One-to-Many Relationships

In a one-to-many relationship, the "one" side references multiple "many" items:

```python
class ParentSQL(EntityBase):
    __tablename__ = 'parent'
    
    # Domain fields
    name = Column(String(255))
    
    # Relationship: one parent to many children
    children = relationship(
        "ChildSQL",
        back_populates="parent",
        lazy="joined",  # Eager loading to prevent N+1 queries
        cascade="all, delete-orphan"  # Cascade deletes to children
    )
    
    def to_entity(self):
        """Convert to domain entity"""
        # Basic field conversion
        base_fields = {
            "ecs_id": UUID(self.ecs_id),
            "lineage_id": UUID(self.lineage_id),
            "parent_id": UUID(self.parent_id) if self.parent_id else None,
            "created_at": self.created_at,
            "old_ids": [UUID(id_str) for id_str in self.old_ids],
            "from_storage": True
        }
        
        # Convert children - important for one-to-many
        children_entities = [child.to_entity() for child in self.children]
        
        return Parent(
            **base_fields,
            name=self.name,
            children=children_entities
        )
    
    def handle_relationships(self, entity, session, orm_objects=None):
        """Handle relationships from entity to SQL model"""
        orm_objects = orm_objects or {}
        
        # Handle children relationship
        self.children = []
        for child in entity.children:
            child_id = str(child.ecs_id)
            if child_id in orm_objects:
                self.children.append(orm_objects[child_id])
```

### 3.2 Many-to-One Relationships

In a many-to-one relationship, multiple "many" items refer to a single "one" item:

```python
class ChildSQL(EntityBase):
    __tablename__ = 'child'
    
    # Domain fields
    name = Column(String(255))
    
    # Foreign key for the parent
    parent_id_fk = Column(Integer, ForeignKey('parent.id'))
    
    # Relationship: many children to one parent
    parent = relationship(
        "ParentSQL",
        back_populates="children",
        lazy="joined"
    )
    
    def to_entity(self):
        """Convert to domain entity"""
        # Basic conversion
        base_fields = {
            "ecs_id": UUID(self.ecs_id),
            "lineage_id": UUID(self.lineage_id),
            "parent_id": UUID(self.parent_id) if self.parent_id else None,
            "created_at": self.created_at,
            "old_ids": [UUID(id_str) for id_str in self.old_ids],
            "from_storage": True
        }
        
        # Convert parent - important for many-to-one
        parent_entity = self.parent.to_entity() if self.parent else None
        
        return Child(
            **base_fields,
            name=self.name,
            parent=parent_entity
        )
    
    def handle_relationships(self, entity, session, orm_objects=None):
        """Handle relationships from entity to SQL model"""
        orm_objects = orm_objects or {}
        
        # Handle parent relationship
        if entity.parent and str(entity.parent.ecs_id) in orm_objects:
            self.parent = orm_objects[str(entity.parent.ecs_id)]
```

### 3.3 Many-to-Many Relationships

Many-to-many relationships use association tables:

```python
# Association table for many-to-many relationship
thread_message_link = Table(
    'thread_message_link', 
    Base.metadata,
    Column('thread_id', Integer, ForeignKey('chat_thread.id'), primary_key=True),
    Column('message_id', Integer, ForeignKey('chat_message.id'), primary_key=True)
)

class ChatThreadSQL(EntityBase):
    __tablename__ = 'chat_thread'
    
    # Domain fields
    name = Column(String(255))
    
    # Many-to-many relationship with messages
    messages = relationship(
        "ChatMessageSQL",
        secondary=thread_message_link,
        back_populates="threads",
        lazy="joined",
        order_by="ChatMessageSQL.timestamp"
    )
    
    def to_entity(self):
        """Convert to domain entity"""
        # Basic conversion
        base_fields = {
            "ecs_id": UUID(self.ecs_id),
            "lineage_id": UUID(self.lineage_id),
            "parent_id": UUID(self.parent_id) if self.parent_id else None,
            "created_at": self.created_at,
            "old_ids": [UUID(id_str) for id_str in self.old_ids],
            "from_storage": True
        }
        
        # Convert messages
        message_entities = [message.to_entity() for message in self.messages]
        
        return ChatThread(
            **base_fields,
            name=self.name,
            history=message_entities  # Note the field name difference
        )
    
    def handle_relationships(self, entity, session, orm_objects=None):
        """Handle relationships from entity to SQL model"""
        orm_objects = orm_objects or {}
        
        # Handle messages relationship (many-to-many)
        self.messages = []
        for message in entity.history:  # Note the field name difference
            message_id = str(message.ecs_id)
            if message_id in orm_objects:
                self.messages.append(orm_objects[message_id])
```

### 3.4 One-to-One Relationships

One-to-one relationships can be implemented with `uselist=False`:

```python
class UserSQL(EntityBase):
    __tablename__ = 'user'
    
    # Domain fields
    username = Column(String(255))
    
    # One-to-one relationship with profile
    profile = relationship(
        "ProfileSQL",
        uselist=False,  # This makes it one-to-one
        back_populates="user",
        lazy="joined",
        cascade="all, delete-orphan"
    )
    
    def to_entity(self):
        """Convert to domain entity"""
        # Basic conversion
        base_fields = {
            "ecs_id": UUID(self.ecs_id),
            "lineage_id": UUID(self.lineage_id),
            "parent_id": UUID(self.parent_id) if self.parent_id else None,
            "created_at": self.created_at,
            "old_ids": [UUID(id_str) for id_str in self.old_ids],
            "from_storage": True
        }
        
        # Convert profile (one-to-one)
        profile_entity = self.profile.to_entity() if self.profile else None
        
        return User(
            **base_fields,
            username=self.username,
            profile=profile_entity
        )
    
    def handle_relationships(self, entity, session, orm_objects=None):
        """Handle relationships from entity to SQL model"""
        orm_objects = orm_objects or {}
        
        # Handle profile relationship (one-to-one)
        if entity.profile and str(entity.profile.ecs_id) in orm_objects:
            self.profile = orm_objects[str(entity.profile.ecs_id)]
```

### 3.5 Association Object Pattern

For many-to-many relationships with extra data:

```python
class MessageUsageSQL(EntityBase):
    """Association object between ChatMessage and Usage"""
    __tablename__ = 'message_usage'
    
    # Foreign keys
    message_id = Column(Integer, ForeignKey('chat_message.id'), primary_key=True)
    usage_id = Column(Integer, ForeignKey('usage.id'), primary_key=True)
    
    # Extra data
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC))
    context_size = Column(Integer)
    
    # Relationships
    message = relationship("ChatMessageSQL", back_populates="usage_associations")
    usage = relationship("UsageSQL", back_populates="message_associations")
    
    def to_entity(self):
        """Convert to domain entity"""
        from your_domain_module import MessageUsage
        
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
    
    def handle_relationships(self, entity, session, orm_objects=None):
        """Handle relationships from entity to SQL model"""
        orm_objects = orm_objects or {}
        
        if entity.message_id and str(entity.message_id) in orm_objects:
            self.message = orm_objects[str(entity.message_id)]
            
        if entity.usage_id and str(entity.usage_id) in orm_objects:
            self.usage = orm_objects[str(entity.usage_id)]
```

### 3.6 Self-Referential Relationships

For entities that reference other entities of the same type:

```python
class CategorySQL(EntityBase):
    __tablename__ = 'category'
    
    # Domain fields
    name = Column(String(255))
    
    # Self-referential relationship
    parent_category_id = Column(Integer, ForeignKey('category.id'))
    parent_category = relationship("CategorySQL", remote_side="CategorySQL.id", backref="subcategories")
    
    def to_entity(self):
        """Convert to domain entity"""
        from your_domain_module import Category
        
        # Basic conversion
        base_fields = {
            "ecs_id": UUID(self.ecs_id),
            "lineage_id": UUID(self.lineage_id),
            "parent_id": UUID(self.parent_id) if self.parent_id else None,
            "created_at": self.created_at,
            "old_ids": [UUID(id_str) for id_str in self.old_ids],
            "from_storage": True
        }
        
        # Convert parent category
        parent_entity = self.parent_category.to_entity() if self.parent_category else None
        
        # Convert subcategories
        subcategory_entities = [subcat.to_entity() for subcat in self.subcategories]
        
        return Category(
            **base_fields,
            name=self.name,
            parent_category=parent_entity,
            subcategories=subcategory_entities
        )
```

## 4. Advanced Relationship Handling Techniques

### 4.1 Two-Phase Loading

For complex object graphs, we use a two-phase approach:

```python
def _store_entity_tree(self, entity, session):
    """Store an entity tree in a single transaction."""
    logger.debug(f"Storing entity tree for {type(entity).__name__}({entity.ecs_id})")
    
    # Phase 1: Collect all entities in the tree
    entities_to_store = {}
    orm_objects = {}
    
    def collect_entities(e):
        entity_id_str = str(e.ecs_id)
        if entity_id_str not in entities_to_store:
            # Create snapshot
            snapshot = create_cold_snapshot(e)
            entities_to_store[entity_id_str] = snapshot
            # Collect nested entities
            for sub in e.get_sub_entities():
                collect_entities(sub)
    
    # Collect all entities in the tree
    collect_entities(entity)
    
    # Phase 2a: Create all ORM objects (without relationships)
    for entity_id_str, e in entities_to_store.items():
        orm_cls = self._get_orm_class(e)
        if not orm_cls:
            logger.error(f"No ORM mapping found for {type(e)}")
            continue
        
        # Check if entity exists
        existing = session.query(orm_cls).filter(orm_cls.ecs_id == entity_id_str).first()
        
        if existing:
            # Update existing record
            orm_objects[entity_id_str] = existing
            # Update fields (excluding relationships)
            for field, value in e.model_dump(exclude={"id"}).items():
                if field not in orm_cls.__mapper__.relationships and hasattr(existing, field):
                    setattr(existing, field, value)
        else:
            # Create new record
            orm_obj = orm_cls.from_entity(e)
            session.add(orm_obj)
            orm_objects[entity_id_str] = orm_obj
    
    # Flush to get IDs assigned
    session.flush()
    
    # Phase 2b: Set up relationships
    for entity_id_str, orm_obj in orm_objects.items():
        entity = entities_to_store[entity_id_str]
        try:
            # Use handle_relationships to set up relationships
            orm_obj.handle_relationships(entity, session, orm_objects)
        except Exception as e:
            logger.error(f"Error in handle_relationships: {str(e)}")
    
    # Refresh objects to ensure changes are loaded
    for orm_obj in orm_objects.values():
        session.refresh(orm_obj)
```

### 4.2 Eager Loading with Joins

To avoid N+1 query problems, use eager loading with joins:

```python
def get_cold_snapshot(self, entity_id: UUID, session=None):
    """Get the stored version of an entity."""
    entity_id_str = str(entity_id)
    
    # Create or reuse session
    use_existing_session = session is not None
    session = session or self._session_factory()
    
    try:
        # Try with known class first
        cls_maybe = self._entity_class_map.get(entity_id_str)
        if cls_maybe:
            orm_cls = self._entity_to_orm_map.get(cls_maybe)
            if orm_cls:
                # Query with eager loading
                query = session.query(orm_cls).filter(orm_cls.ecs_id == entity_id_str)
                
                # Add eager loading for all relationships
                for relationship_name in orm_cls.__mapper__.relationships.keys():
                    query = query.options(joinedload(getattr(orm_cls, relationship_name)))
                
                result = query.first()
                if result:
                    return result.to_entity()
        
        # Fallback: scan all tables
        for entity_cls, orm_cls in self._entity_to_orm_map.items():
            query = session.query(orm_cls).filter(orm_cls.ecs_id == entity_id_str)
            
            # Add eager loading for all relationships
            for relationship_name in orm_cls.__mapper__.relationships.keys():
                query = query.options(joinedload(getattr(orm_cls, relationship_name)))
            
            result = query.first()
            if result:
                # Cache the class for future lookups
                self._entity_class_map[entity_id_str] = type(result.to_entity())
                return result.to_entity()
        
        return None
    finally:
        # Close session if we created it
        if not use_existing_session:
            session.close()
```

### 4.3 Relationship Loading Order

When dealing with complex hierarchies, loading order matters:

```python
def handle_complex_hierarchy(self, entity, session, orm_objects=None):
    """Handle relationships in a specific order to avoid dependency issues."""
    orm_objects = orm_objects or {}
    
    # 1. First handle direct references (foreign keys)
    if entity.owner and str(entity.owner.ecs_id) in orm_objects:
        self.owner = orm_objects[str(entity.owner.ecs_id)]
    
    # 2. Then handle child collections
    self.items = []
    for item in entity.items:
        item_id = str(item.ecs_id)
        if item_id in orm_objects:
            self.items.append(orm_objects[item_id])
    
    # 3. Finally handle many-to-many associations
    self.tags = []
    for tag in entity.tags:
        tag_id = str(tag.ecs_id)
        if tag_id in orm_objects:
            self.tags.append(orm_objects[tag_id])
```

### 4.4 Polymorphic Relationships

For handling different entity types in the same relationship:

```python
class ContentSQL(EntityBase):
    """Base class for different content types"""
    __tablename__ = 'content'
    
    content_type = Column(String(20), nullable=False)
    title = Column(String(255))
    
    __mapper_args__ = {
        'polymorphic_on': content_type,
        'polymorphic_identity': 'base'
    }

class TextContentSQL(ContentSQL):
    """Text content implementation"""
    text = Column(Text)
    
    __mapper_args__ = {
        'polymorphic_identity': 'text'
    }
    
    def to_entity(self):
        """Convert to appropriate domain entity"""
        from your_domain_module import TextContent
        
        # Basic conversion
        base_fields = {
            "ecs_id": UUID(self.ecs_id),
            "lineage_id": UUID(self.lineage_id),
            "parent_id": UUID(self.parent_id) if self.parent_id else None,
            "created_at": self.created_at,
            "old_ids": [UUID(id_str) for id_str in self.old_ids],
            "from_storage": True
        }
        
        return TextContent(
            **base_fields,
            title=self.title,
            text=self.text
        )

class ImageContentSQL(ContentSQL):
    """Image content implementation"""
    image_url = Column(String(255))
    
    __mapper_args__ = {
        'polymorphic_identity': 'image'
    }
    
    def to_entity(self):
        """Convert to appropriate domain entity"""
        from your_domain_module import ImageContent
        
        # Basic conversion
        base_fields = {
            "ecs_id": UUID(self.ecs_id),
            "lineage_id": UUID(self.lineage_id),
            "parent_id": UUID(self.parent_id) if self.parent_id else None,
            "created_at": self.created_at,
            "old_ids": [UUID(id_str) for id_str in self.old_ids],
            "from_storage": True
        }
        
        return ImageContent(
            **base_fields,
            title=self.title,
            image_url=self.image_url
        )
```

## 5. Entity Registry and Storage Interface

The `SqlEntityStorage` class ties everything together:

```python
class SqlEntityStorage:
    """
    SQLAlchemy-based storage implementation for versioned entities.
    Handles conversion between domain entities and SQL models.
    """
    def __init__(self, session_factory, entity_to_orm_map):
        self._logger = logging.getLogger("SqlEntityStorage")
        self._session_factory = session_factory
        self._entity_to_orm_map = entity_to_orm_map
        self._entity_class_map = {}  # Cache for entity class lookups
    
    def get_session(self, existing_session=None):
        """Get a session - either the provided one or a new one."""
        if existing_session is not None:
            return existing_session, False
        return self._session_factory(), True
    
    def has_entity(self, entity_id, session=None):
        """Check if an entity exists in storage."""
        entity_id_str = str(entity_id)
        
        # Check cache first
        if entity_id_str in self._entity_class_map:
            return True
        
        # Query database
        session_obj, should_close = self.get_session(session)
        try:
            for entity_cls, orm_cls in self._entity_to_orm_map.items():
                result = session_obj.query(orm_cls).filter(orm_cls.ecs_id == entity_id_str).first()
                if result is not None:
                    # Cache the class for future lookups
                    entity = result.to_entity()
                    self._entity_class_map[entity_id_str] = type(entity)
                    return True
            return False
        finally:
            if should_close:
                session_obj.close()
    
    def register(self, entity_or_id, session=None):
        """Register an entity or get it by ID."""
        if isinstance(entity_or_id, UUID):
            return self.get(entity_or_id, None, session=session)
        
        entity = entity_or_id
        self._logger.info(f"Registering entity {type(entity).__name__}({entity.ecs_id})")
        
        # Create or reuse session
        own_session = session is None
        session_obj = session or self._session_factory()
        
        try:
            # Store the entity tree
            self._store_entity_tree(entity, session_obj)
            
            # Commit if we created the session
            if own_session:
                session_obj.commit()
            
            return entity
        except Exception as e:
            if own_session:
                self._logger.error(f"Error registering entity, rolling back: {str(e)}")
                session_obj.rollback()
            return None
        finally:
            if own_session:
                session_obj.close()
    
    def get(self, entity_id, expected_type=None, session=None):
        """Get an entity by ID with optional type checking."""
        entity = self.get_cold_snapshot(entity_id, session)
        
        if not entity:
            return None
        
        if expected_type and not isinstance(entity, expected_type):
            self._logger.error(f"Type mismatch: got {type(entity).__name__}, expected {expected_type.__name__}")
            return None
        
        # Create a warm copy
        warm_copy = deepcopy(entity)
        warm_copy.live_id = uuid4()  # Assign a new live_id
        warm_copy.from_storage = True
        
        return warm_copy
```

## 6. Best Practices for SQLAlchemy in a Versioned System

### 6.1 Consistent UUID Handling

Always convert UUIDs to strings in database storage:

```python
# In SQLAlchemy models
ecs_id = Column(String(36), index=True)  # Store as string

# In from_entity
sql_obj.ecs_id = str(entity.ecs_id)  # Convert UUID to string

# In to_entity
entity_obj.ecs_id = UUID(self.ecs_id)  # Convert string to UUID
```

### 6.2 Proper Relationship Cascades

Configure cascades to match your domain logic:

```python
# Parent deletes cascade to children
children = relationship("ChildSQL", cascade="all, delete-orphan")

# Many-to-many without cascading deletes
tags = relationship("TagSQL", secondary=item_tag_link, cascade="save-update")

# One-to-one with bidirectional cascades
profile = relationship("ProfileSQL", uselist=False, cascade="all, delete-orphan")
```

### 6.3 Eager Loading Configuration

Use eager loading to avoid N+1 query problems:

```python
# Always eager load critical relationships
messages = relationship("MessageSQL", lazy="joined")

# Selectively eager load based on use case
attachments = relationship("AttachmentSQL", lazy="selectin")

# Explicitly join in queries for ad-hoc eager loading
query = session.query(ThreadSQL).options(
    joinedload(ThreadSQL.messages),
    joinedload(ThreadSQL.messages).joinedload(MessageSQL.author)
)
```

### 6.4 Handling Circular Dependencies

For circular dependencies, use string names and late binding:

```python
class UserSQL(EntityBase):
    threads = relationship("ThreadSQL", back_populates="owner")

class ThreadSQL(EntityBase):
    owner = relationship("UserSQL", back_populates="threads")
```

### 6.5 Error Handling and Debugging

Include robust error handling in relationship management:

```python
def handle_relationships(self, entity, session, orm_objects=None):
    """Handle relationships with proper error handling."""
    try:
        orm_objects = orm_objects or {}
        
        # Handle relationships
        if entity.parent and str(entity.parent.ecs_id) in orm_objects:
            self.parent = orm_objects[str(entity.parent.ecs_id)]
        
        # Handle children
        self.children = []
        for child in entity.children:
            child_id = str(child.ecs_id)
            if child_id in orm_objects:
                self.children.append(orm_objects[child_id])
    except Exception as e:
        logger = logging.getLogger(self.__class__.__name__)
        logger.error(f"Error handling relationships: {str(e)}")
        logger.debug("Traceback:", exc_info=True)
        raise
```

## 7. Putting It All Together: A Complete Example

Let's put all these concepts together in a complete example with a ChatThread and ChatMessage:

```python
# Association table for thread-message many-to-many relationship
thread_message_link = Table(
    'thread_message_link', 
    Base.metadata,
    Column('thread_id', Integer, ForeignKey('chat_thread.id'), primary_key=True),
    Column('message_id', Integer, ForeignKey('chat_message.id'), primary_key=True)
)

class ChatMessageSQL(EntityBase):
    """SQL model for ChatMessage"""
    __tablename__ = 'chat_message'
    
    # Domain fields
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC))
    parent_message_uuid = Column(String(36))
    
    # Relationships
    threads = relationship(
        "ChatThreadSQL",
        secondary=thread_message_link,
        back_populates="messages",
        lazy="joined"
    )
    
    def to_entity(self):
        """Convert to domain entity"""
        from minference.threads.models import ChatMessage, MessageRole
        
        # Basic conversion
        base_fields = {
            "ecs_id": UUID(self.ecs_id),
            "lineage_id": UUID(self.lineage_id),
            "parent_id": UUID(self.parent_id) if self.parent_id else None,
            "created_at": self.created_at,
            "old_ids": [UUID(id_str) for id_str in self.old_ids],
            "from_storage": True
        }
        
        return ChatMessage(
            **base_fields,
            role=MessageRole(self.role),
            content=self.content,
            timestamp=self.timestamp,
            parent_message_uuid=UUID(self.parent_message_uuid) if self.parent_message_uuid else None
        )
    
    @classmethod
    def from_entity(cls, entity):
        """Create from domain entity"""
        from minference.threads.models import ChatMessage
        
        # Type check
        if not isinstance(entity, ChatMessage):
            raise TypeError(f"Expected ChatMessage, got {type(entity).__name__}")
        
        # Convert fields
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
    
    def handle_relationships(self, entity, session, orm_objects=None):
        """Handle relationships"""
        # ChatMessage doesn't need to handle relationships in this example
        pass

class ChatThreadSQL(EntityBase):
    """SQL model for ChatThread"""
    __tablename__ = 'chat_thread'
    
    # Domain fields
    name = Column(String(255))
    use_history = Column(Boolean, default=True)
    
    # Relationships
    messages = relationship(
        "ChatMessageSQL",
        secondary=thread_message_link,
        back_populates="threads",
        lazy="joined",
        order_by="ChatMessageSQL.timestamp"
    )
    
    def to_entity(self):
        """Convert to domain entity"""
        from minference.threads.models import ChatThread
        
        # Basic conversion
        base_fields = {
            "ecs_id": UUID(self.ecs_id),
            "lineage_id": UUID(self.lineage_id),
            "parent_id": UUID(self.parent_id) if self.parent_id else None,
            "created_at": self.created_at,
            "old_ids": [UUID(id_str) for id_str in self.old_ids],
            "from_storage": True
        }
        
        # Convert messages
        message_entities = [message.to_entity() for message in self.messages]
        
        return ChatThread(
            **base_fields,
            name=self.name,
            history=message_entities,  # Note field name difference
            use_history=self.use_history
        )
    
    @classmethod
    def from_entity(cls, entity):
        """Create from domain entity"""
        from minference.threads.models import ChatThread
        
        # Type check
        if not isinstance(entity, ChatThread):
            raise TypeError(f"Expected ChatThread, got {type(entity).__name__}")
        
        # Convert fields
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
        """Handle relationships"""
        try:
            orm_objects = orm_objects or {}
            
            # Handle messages relationship (many-to-many)
            self.messages = []
            
            for message in entity.history:
                message_id = str(message.ecs_id)
                if message_id in orm_objects:
                    self.messages.append(orm_objects[message_id])
                else:
                    logger = logging.getLogger("ChatThreadSQL")
                    logger.warning(f"Message {message_id} not found in orm_objects")
        except Exception as e:
            logger = logging.getLogger("ChatThreadSQL")
            logger.error(f"Error handling relationships: {str(e)}")
            logger.debug("Traceback:", exc_info=True)
            raise
```

## 8. Using the SQL Storage Implementation

Finally, let's see how to set up and use the SQL storage:

```python
# Set up the SQLAlchemy engine and session factory
def setup_sql_storage(connection_string):
    """Set up SQL storage for EntityRegistry"""
    # Create engine
    engine = create_engine(connection_string)
    
    # Create tables
    Base.metadata.create_all(engine)
    
    # Create session factory
    session_factory = sessionmaker(bind=engine)
    
    # Create entity to ORM mapping
    from minference.threads.models import ChatThread, ChatMessage
    entity_orm_map = {
        ChatThread: ChatThreadSQL,
        ChatMessage: ChatMessageSQL,
        # Add other mappings here
    }
    
    # Create storage
    storage = SqlEntityStorage(session_factory, entity_orm_map)
    
    # Configure EntityRegistry to use this storage
    from minference.ecs.entity import EntityRegistry
    EntityRegistry.use_storage(storage)
    
    return storage

# Example usage
if __name__ == "__main__":
    # Set up storage with SQLite for testing
    storage = setup_sql_storage("sqlite:///test.db")
    
    # Create a chat thread with messages
    from minference.threads.models import ChatThread, ChatMessage, MessageRole
    
    thread = ChatThread(name="Test Thread")
    
    message1 = ChatMessage(
        role=MessageRole.user,
        content="Hello!",
        chat_thread_id=thread.ecs_id
    )
    
    message2 = ChatMessage(
        role=MessageRole.assistant,
        content="Hi there!",
        chat_thread_id=thread.ecs_id,
        parent_message_uuid=message1.ecs_id
    )
    
    thread.history.append(message1)
    thread.history.append(message2)
    
    # Entities are automatically registered due to Pydantic's model_validator
    
    # Retrieve the thread
    retrieved_thread = ChatThread.get(thread.ecs_id)
    
    # Print messages
    for message in retrieved_thread.history:
        print(f"{message.role}: {message.content}")
```

This guide provides a comprehensive overview of implementing versioned entities with SQLAlchemy, focusing on entity conversion and relationship management. By following these patterns, you can create a robust system that maintains version history while leveraging the power of SQLAlchemy's ORM capabilities.