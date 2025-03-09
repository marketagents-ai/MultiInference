# SQLAlchemy Implementation Plan for Entity Component System

## Background

The current SQL backend implementation uses SQLModel, which causes several issues:
1. Confusing syntax (mix of SQLAlchemy and Pydantic features)
2. Complex relationship handling
3. Poor compatibility with LLM code generation
4. Versioning complexity

Based on our testing of the in-memory implementation, we've identified key points for implementation.

## Implementation Strategy

### 1. Database Models

Replace SQLModel with pure SQLAlchemy ORM, using declarative syntax:

```python
class BaseEntitySQL(Base):
    """Base class for all entity SQL models."""
    __tablename__ = 'base_entities'
    
    # Primary key (auto-increment ID for database)
    id = Column(Integer, primary_key=True)
    
    # Entity versioning fields
    ecs_id = Column(String, index=True)  # Store as string for better compatibility
    lineage_id = Column(String, index=True)
    parent_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    old_ids = Column(JSON, default=list)
    
    # Type discriminator for polymorphic models
    type = Column(String)
    
    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'base_entity'
    }
    
    def to_entity(self) -> Entity:
        """Convert SQL model to Entity object."""
        # Implementation depends on specific entity type
        raise NotImplementedError
    
    @classmethod
    def from_entity(cls, entity: Entity) -> 'BaseEntitySQL':
        """Convert Entity object to SQL model."""
        # Implementation depends on specific entity type
        raise NotImplementedError
```

### 2. Relationship Handling

Use ID references for relationships to avoid circular reference issues:

```python
class ChatThreadSQL(BaseEntitySQL):
    """SQL model for ChatThread entity."""
    __tablename__ = 'chat_threads'
    
    # Link to parent class
    id = Column(Integer, ForeignKey('base_entities.id'), primary_key=True)
    
    # Thread properties
    title = Column(String)
    
    # ID references for relationships
    message_ids = Column(JSON, default=list)  # Store UUIDs as strings
    
    __mapper_args__ = {
        'polymorphic_identity': 'chat_thread'
    }
    
    def to_entity(self) -> 'ChatThread':
        """Convert to Entity object."""
        from minference.threads.models import ChatThread
        
        # Basic fields
        entity = ChatThread(
            ecs_id=UUID(self.ecs_id),
            lineage_id=UUID(self.lineage_id),
            title=self.title,
            from_storage=True
        )
        
        # Set parent ID if exists
        if self.parent_id:
            entity.parent_id = UUID(self.parent_id)
            
        # Convert old_ids from strings to UUIDs
        entity.old_ids = [UUID(old_id) for old_id in self.old_ids]
        
        return entity
    
    @classmethod
    def from_entity(cls, entity: 'ChatThread') -> 'ChatThreadSQL':
        """Convert Entity to SQL model."""
        return ChatThreadSQL(
            ecs_id=str(entity.ecs_id),
            lineage_id=str(entity.lineage_id),
            parent_id=str(entity.parent_id) if entity.parent_id else None,
            old_ids=[str(id) for id in entity.old_ids],
            title=entity.title,
            message_ids=[str(message.ecs_id) for message in entity.messages]
        )
```

### 3. Entity Storage Implementation

Create a SQLAlchemy implementation of `EntityStorage`:

```python
class SqlAlchemyEntityStorage(EntityStorage):
    """SQLAlchemy-based entity storage."""
    
    def __init__(self, engine_url: str, entity_mapping: Dict[Type[Entity], Type[BaseEntitySQL]]):
        """Initialize storage with database connection."""
        self.engine = create_engine(engine_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.entity_mapping = entity_mapping
    
    def register(self, entity_or_id: Union[Entity, UUID]) -> Optional[Entity]:
        """Register an entity in SQL storage."""
        if isinstance(entity_or_id, UUID):
            return self.get(entity_or_id)
            
        entity = entity_or_id
        with self.Session() as session:
            # Check if entity exists
            if self.has_entity(entity.ecs_id):
                existing = self.get_cold_snapshot(entity.ecs_id)
                if existing and entity.has_modifications(existing):
                    # Fork if needed
                    entity = entity.fork()
            
            # Map entity to SQL model
            sql_model_class = self._get_sql_model_class(entity)
            if not sql_model_class:
                return None
                
            # Convert entity to SQL model
            sql_model = sql_model_class.from_entity(entity)
            
            # Save to database
            session.add(sql_model)
            session.commit()
            
            return entity
    
    def get_cold_snapshot(self, entity_id: UUID) -> Optional[Entity]:
        """Get entity from database."""
        with self.Session() as session:
            # Try all entity types until we find a match
            for entity_type, sql_model_class in self.entity_mapping.items():
                stmt = select(sql_model_class).where(sql_model_class.ecs_id == str(entity_id))
                result = session.execute(stmt).scalar_one_or_none()
                if result:
                    return result.to_entity()
        return None
    
    # ... other methods similarly implemented
```

### 4. Testing Approach

1. Create a SQLite in-memory database for testing
2. Define test entity models with relationships
3. Test all the same operations from in-memory implementation
4. Ensure all relationship patterns work (one-to-many, many-to-many, hierarchical)

## Key Improvements over Current Implementation

1. **Clearer Separation of Concerns**:
   - SQL models handle database representation
   - Entity classes handle domain logic
   - No confusion between SQL and domain models

2. **Explicit ID References**:
   - Avoids circular reference issues
   - Provides clearer code boundaries
   - Makes relationships more maintainable

3. **Modern SQLAlchemy Patterns**:
   - Uses SQLAlchemy 2.0-compatible syntax
   - Leverages Core expressions for queries
   - Uses proper relationship loading strategies

4. **Simplified Entity Conversion**:
   - Clear `to_entity` and `from_entity` methods
   - Explicit UUID conversions
   - Better handling of nested objects

5. **Improved Type Safety**:
   - Strong typing throughout implementation
   - Explicit null handling
   - Better generics support

## Implementation Timeline

1. **Phase 1**: Create base SQLAlchemy models and conversions
2. **Phase 2**: Implement storage class with basic operations
3. **Phase 3**: Add relationship handling for all patterns
4. **Phase 4**: Write comprehensive tests for all functionality
5. **Phase 5**: Migrate existing code to new implementation