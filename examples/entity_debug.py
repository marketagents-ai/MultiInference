#!/usr/bin/env python
"""
Comprehensive test script for the entity.py refactoring.
"""

import sys
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Literal, Any, TypeVar, Type, cast, Protocol, runtime_checkable
from uuid import UUID, uuid4
import json

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import from our refactored entity.py
from minference.ecs.entity import (
    Entity, EntityRegistry, EntityDiff, 
    InMemoryEntityStorage, SqlEntityStorage,
    entity_tracer, compare_entity_fields,
    SQLModelType
)

from pydantic import Field

# Initialize storage - start with in-memory
EntityRegistry.use_storage(InMemoryEntityStorage())

#############################################################################
# Type Protocols for proper static type checking
#############################################################################

@runtime_checkable
class AddressProtocol(Protocol):
    street: str
    city: str
    zipcode: str
    country: str
    id: UUID

@runtime_checkable
class PersonProtocol(Protocol):
    name: str
    age: int
    email: Optional[str]
    address: Optional['AddressProtocol']
    id: UUID

@runtime_checkable
class TagProtocol(Protocol):
    name: str
    color: str
    id: UUID

@runtime_checkable
class TaskProtocol(Protocol):
    title: str
    description: Optional[str]
    status: Literal["pending", "in_progress", "completed"]
    priority: int
    tags: List['TagProtocol']
    id: UUID

@runtime_checkable
class CommentProtocol(Protocol):
    content: str
    created_at: datetime
    author: Optional['PersonProtocol']
    id: UUID

@runtime_checkable
class ProjectProtocol(Protocol):
    name: str
    description: str
    tasks: List['TaskProtocol']
    members: List['PersonProtocol']
    owner: Optional['PersonProtocol']
    tags: List['TagProtocol']
    comments: List['CommentProtocol']
    id: UUID

# Type variables for specific entity types
T_SimpleEntity = TypeVar('T_SimpleEntity', bound='SimpleEntity')
T_Address = TypeVar('T_Address', bound='Address')
T_Person = TypeVar('T_Person', bound='Person')
T_Tag = TypeVar('T_Tag', bound='Tag')
T_Task = TypeVar('T_Task', bound='Task')
T_Project = TypeVar('T_Project', bound='Project')

#############################################################################
# Define test entities representing different relationship patterns
#############################################################################

class SimpleEntity(Entity):
    """A basic entity with simple fields."""
    name: str
    value: int = 0
    description: Optional[str] = None


class Address(Entity):
    """Component entity for nesting."""
    street: str
    city: str
    zipcode: str
    country: str = "USA"


class Person(Entity):
    """Entity with nested entity (one-to-one)."""
    name: str
    age: int
    email: Optional[str] = None
    address: Optional[Address] = None


class Tag(Entity):
    """Simple tagging entity for many-to-many relationships."""
    name: str
    color: str = "#000000"


class Task(Entity):
    """Entity with a one-to-many relationship."""
    title: str
    description: Optional[str] = None
    status: Literal["pending", "in_progress", "completed"] = "pending"
    priority: int = 1
    tags: List[Tag] = Field(default_factory=list)


class Comment(Entity):
    """Entity that can reference another entity."""
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    author: Optional[Person] = None


class Project(Entity):
    """Complex entity with multiple nested entities and relationships."""
    name: str
    description: str = ""
    tasks: List[Task] = Field(default_factory=list)
    members: List[Person] = Field(default_factory=list)
    owner: Optional[Person] = None
    tags: List[Tag] = Field(default_factory=list)
    comments: List[Comment] = Field(default_factory=list)


#############################################################################
# Type guards for entity typechecking
#############################################################################

def is_address(entity: Entity) -> bool:
    """Type guard for Address entities"""
    return isinstance(entity, Address)

def is_person(entity: Entity) -> bool:
    """Type guard for Person entities"""
    return isinstance(entity, Person)

def is_tag(entity: Entity) -> bool:
    """Type guard for Tag entities"""
    return isinstance(entity, Tag)

def is_task(entity: Entity) -> bool:
    """Type guard for Task entities"""
    return isinstance(entity, Task)

def is_project(entity: Entity) -> bool:
    """Type guard for Project entities"""
    return isinstance(entity, Project)

# Type assertion helpers
def as_address(entity: Entity) -> Address:
    """Assert entity is an Address"""
    assert is_address(entity)
    return cast(Address, entity)

def as_person(entity: Entity) -> Person:
    """Assert entity is a Person"""
    assert is_person(entity)
    return cast(Person, entity)

def as_tag(entity: Entity) -> Tag:
    """Assert entity is a Tag"""
    assert is_tag(entity)
    return cast(Tag, entity)

def as_task(entity: Entity) -> Task:
    """Assert entity is a Task"""
    assert is_task(entity)
    return cast(Task, entity)

def as_project(entity: Entity) -> Project:
    """Assert entity is a Project"""
    assert is_project(entity)
    return cast(Project, entity)

#############################################################################
# Define SQL model counterparts for the test entities
#############################################################################

from sqlmodel import SQLModel, Field as SQLField, Column, JSON, Session, create_engine, Relationship

class SimpleEntitySQL(SQLModel, table=True):
    """SQL model for SimpleEntity"""
    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = SQLField(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = SQLField(default=None)
    
    name: str
    value: int
    description: Optional[str] = None
    
    def to_entity(self) -> SimpleEntity:
        return SimpleEntity(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            name=self.name,
            value=self.value,
            description=self.description,
            from_storage=True
        )
    
    @classmethod
    def from_entity(cls, entity: SimpleEntity) -> "SimpleEntitySQL":
        return cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            name=entity.name,
            value=entity.value,
            description=entity.description
        )


class AddressSQL(SQLModel, table=True):
    """SQL model for Address"""
    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = SQLField(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = SQLField(default=None)
    
    street: str
    city: str
    zipcode: str
    country: str
    
    def to_entity(self) -> Address:
        return Address(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            street=self.street,
            city=self.city,
            zipcode=self.zipcode,
            country=self.country,
            from_storage=True
        )
    
    @classmethod
    def from_entity(cls, entity: Address) -> "AddressSQL":
        return cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            street=entity.street,
            city=entity.city,
            zipcode=entity.zipcode,
            country=entity.country
        )


class PersonSQL(SQLModel, table=True):
    """SQL model for Person with nested address reference"""
    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = SQLField(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = SQLField(default=None)
    
    name: str
    age: int
    email: Optional[str] = None
    address_id: Optional[UUID] = None
    
    def to_entity(self) -> Person:
        address = None
        if self.address_id:
            # Use a global engine or session factory
            with Session(engine) as session:
                address_sql = session.get(AddressSQL, self.address_id)
                if address_sql:
                    address = address_sql.to_entity()
        
        return Person(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            name=self.name,
            age=self.age,
            email=self.email,
            address=address,
            from_storage=True
        )
    
    @classmethod
    def from_entity(cls, entity: Person) -> "PersonSQL":
        return cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            name=entity.name,
            age=entity.age,
            email=entity.email,
            address_id=entity.address.id if entity.address else None
        )


class TagSQL(SQLModel, table=True):
    """SQL model for Tag"""
    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = SQLField(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = SQLField(default=None)
    
    name: str
    color: str
    
    def to_entity(self) -> Tag:
        return Tag(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            name=self.name,
            color=self.color,
            from_storage=True
        )
    
    @classmethod
    def from_entity(cls, entity: Tag) -> "TagSQL":
        return cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            name=entity.name,
            color=entity.color
        )


# Bridge tables for many-to-many relationships
class TaskTagLink(SQLModel, table=True):
    task_id: UUID = SQLField(foreign_key="tasksql.id", primary_key=True)
    tag_id: UUID = SQLField(foreign_key="tagsql.id", primary_key=True)


class TaskSQL(SQLModel, table=True):
    """SQL model for Task with tags many-to-many"""
    id: UUID = SQLField(default_factory=uuid4, primary_key=True)
    lineage_id: UUID = SQLField(default_factory=uuid4, index=True)
    parent_id: Optional[UUID] = SQLField(default=None)
    
    title: str
    description: Optional[str] = None
    status: str  # Enum as string
    priority: int
    
    # Many-to-many relationship with tags through SQLModel
    tags: List["TagSQL"] = Relationship(link_model=TaskTagLink)
    
    def to_entity(self) -> Task:
        return Task(
            id=self.id,
            lineage_id=self.lineage_id,
            parent_id=self.parent_id,
            title=self.title,
            description=self.description,
            status=cast(Literal["pending", "in_progress", "completed"], self.status),
            priority=self.priority,
            tags=[tag.to_entity() for tag in self.tags],
            from_storage=True
        )
    
    @classmethod
    def from_entity(cls, entity: Task) -> "TaskSQL":
        task_sql = cls(
            id=entity.id,
            lineage_id=entity.lineage_id,
            parent_id=entity.parent_id,
            title=entity.title,
            description=entity.description,
            status=entity.status,
            priority=entity.priority
        )
        
        # Tags would be linked separately through the many-to-many link table
        return task_sql


#############################################################################
# Setup SQL storage for testing
#############################################################################

# Define global engine for use throughout tests
engine = create_engine("sqlite:///:memory:", echo=False)

def setup_sql_storage():
    """Set up SQL storage with a test database"""
    # Create all tables
    SQLModel.metadata.create_all(engine)
    
    # Create a session factory
    def session_factory():
        return Session(engine)
    
    # Create entity to ORM mapping with explicit casting for type safety
    entity_to_orm_map: Dict[Type[Entity], Type[SQLModelType]] = {
        SimpleEntity: cast(Type[SQLModelType], SimpleEntitySQL),
        Address: cast(Type[SQLModelType], AddressSQL),
        Person: cast(Type[SQLModelType], PersonSQL),
        Tag: cast(Type[SQLModelType], TagSQL),
        Task: cast(Type[SQLModelType], TaskSQL),
    }
    
    # Create SQL storage
    sql_storage = SqlEntityStorage(
        session_factory=session_factory,
        entity_to_orm_map=entity_to_orm_map
    )
    
    # Use the SQL storage with the registry
    EntityRegistry.use_storage(sql_storage)
    return session_factory


#############################################################################
# Helper Functions
#############################################################################

def print_entity_details(entity: Entity, prefix: str = ""):
    """Print detailed information about an entity"""
    print(f"{prefix}Entity ID: {entity.id}")
    print(f"{prefix}Lineage:   {entity.lineage_id}")
    print(f"{prefix}Parent:    {entity.parent_id or 'None'}")
    print(f"{prefix}Type:      {type(entity).__name__}")
    
    # Print key fields
    for field, value in entity.model_dump().items():
        if field not in {'id', 'lineage_id', 'parent_id', 'live_id', 'old_ids', 'from_storage', 'created_at'}:
            if isinstance(value, Entity):
                nested_id = value.id if value else None
                print(f"{prefix}{field}: {type(value).__name__} (ID: {nested_id})")
            elif isinstance(value, list) and all(isinstance(item, Entity) for item in value):
                print(f"{prefix}{field}: List[{len(value)} entities]")
            else:
                print(f"{prefix}{field}: {value}")
    print()


def print_lineage(entity: Entity):
    """Print the lineage tree for an entity"""
    lineage_id = entity.lineage_id
    mermaid = EntityRegistry.get_lineage_mermaid(lineage_id)
    print(f"Lineage for {type(entity).__name__} ({entity.id}):")
    print(mermaid)


def print_test_header(title: str):
    """Print a formatted test header"""
    print("\n" + "=" * 80)
    print(f" TEST: {title} ".center(80, "="))
    print("=" * 80 + "\n")


def track_modifications(original: Entity, modified: Entity) -> bool:
    """Track modifications between entities and print details"""
    has_mods, diffs = compare_entity_fields(modified, original)
    
    if has_mods:
        print(f"✓ Detected modifications in {type(original).__name__}:")
        for field, diff_info in diffs.items():
            diff_type = diff_info.get("type", "unknown")
            old_val = diff_info.get("old", "N/A")
            new_val = diff_info.get("new", "N/A")
            
            if diff_type == "modified":
                print(f"  - {field}: Changed from '{old_val}' to '{new_val}'")
            elif diff_type == "added":
                print(f"  - {field}: Added with value '{new_val}'")
            elif diff_type == "removed":
                print(f"  - {field}: Removed (was '{old_val}')")
            else:
                print(f"  - {field}: {diff_type}")
    else:
        print(f"✗ No modifications detected in {type(original).__name__}")
    
    return has_mods


@entity_tracer
def modify_entity(entity: SimpleEntity, **kwargs) -> SimpleEntity:
    """Helper to modify an entity while ensuring tracking"""
    for k, v in kwargs.items():
        setattr(entity, k, v)
    return entity


#############################################################################
# Test Cases
#############################################################################

def test_simple_entity_forking():
    """Test creating and forking simple entities"""
    print_test_header("Simple Entity Creation and Forking")
    
    # Create a simple entity
    entity1 = SimpleEntity(name="Test Entity", value=42)
    print("Original entity created:")
    print_entity_details(entity1)
    
    # Retrieve from registry to get a cold snapshot
    stored = EntityRegistry.get(entity1.id)
    if stored:
        print("Retrieved from registry:")
        print_entity_details(stored)
    else:
        print("ERROR: Entity not found in registry")
    
    # Check no modifications yet
    if stored:
        track_modifications(stored, entity1)
    
    # Modify and check for detection
    entity1.value = 100
    print("\nAfter modification:")
    print_entity_details(entity1)
    
    # Check if modifications detected
    if stored:
        modified = track_modifications(stored, entity1)
    
    # Fork the entity
    print("\nForking entity...")
    forked = entity1.fork()
    print("Forked entity:")
    print_entity_details(forked)
    
    # Print the lineage
    print_lineage(forked)


def test_nested_entity_forking():
    """Test forking behavior with nested entities"""
    print_test_header("Nested Entity Relationships and Bottom-up Forking")
    
    # Create a nested structure
    address = Address(street="123 Main St", city="Anytown", zipcode="12345")
    person = Person(name="John Doe", age=30, address=address)
    
    print("Original nested structure:")
    print_entity_details(person)
    
    # Type checking: ensure address exists before using it
    if person.address:
        print_entity_details(person.address, prefix="Address: ")
    
    # Modify the nested entity with proper type assertion
    print("\nModifying nested address...")
    if person.address:
        person_addr = as_address(person.address)
        person_addr.street = "456 New Street"
    
    # Retrieve the original from registry and check for modifications
    stored_person = EntityRegistry.get(person.id)
    stored_address = EntityRegistry.get(address.id)
    
    print("\nChecking address modifications:")
    if stored_address and person.address:
        addr_modified = track_modifications(stored_address, person.address)
    
    print("\nChecking person modifications (should detect nested change):")
    if stored_person:
        person_modified = track_modifications(stored_person, person)
    
    # Fork and check the result
    print("\nForking with nested modifications...")
    forked_person = person.fork()
    
    # Use type assertion for type safety
    forked_person_typed = as_person(forked_person)
    
    print("\nForked structure:")
    print_entity_details(forked_person_typed)
    
    # Type checking: ensure address exists in forked person
    if forked_person_typed.address:
        forked_addr = as_address(forked_person_typed.address)
        print_entity_details(forked_addr, prefix="Address: ")
    
    # Verify IDs have changed appropriately
    print("\nVerifying ID changes:")
    print(f"Original person ID: {person.id}")
    print(f"Forked person ID: {forked_person_typed.id}")
    print(f"Original address ID: {address.id}")
    if forked_person_typed.address:
        print(f"Forked address ID: {forked_person_typed.address.id}")
    
    # Print the lineage
    print_lineage(forked_person_typed)
    if forked_person_typed.address:
        print_lineage(forked_person_typed.address)


def test_one_to_many_forking():
    """Test forking behavior with one-to-many relationships"""
    print_test_header("One-to-Many Relationship Forking")
    
    # Create tags
    tag1 = Tag(name="important", color="#FF0000")
    tag2 = Tag(name="bug", color="#0000FF")
    tag3 = Tag(name="feature", color="#00FF00")
    
    # Create task with tags
    task = Task(
        title="Implement entity forking",
        description="Create a robust forking mechanism for entities",
        status="in_progress",
        priority=1,
        tags=[tag1, tag2, tag3]
    )
    
    # Use type assertion for type safety
    task_typed = as_task(task)
    
    print("Original task with tags:")
    print_entity_details(task_typed)
    print("Tags:")
    for tag in task_typed.tags:
        tag_typed = as_tag(tag)
        print_entity_details(tag_typed, prefix="  ")
    
    # Modify one of the tags with proper type safety
    print("\nModifying a tag...")
    if len(task_typed.tags) > 1:
        tag2_typed = as_tag(task_typed.tags[1])
        tag2_typed.color = "#9900FF"
    
    # Check for modifications in the tag
    stored_tag = EntityRegistry.get(tag2.id)
    print("\nChecking tag modifications:")
    if stored_tag and len(task_typed.tags) > 1:
        tag_modified = track_modifications(stored_tag, task_typed.tags[1])
    
    # Fork the task and check the result
    print("\nForking task...")
    forked_task = task.fork()
    
    # Use type assertion for type safety
    forked_task_typed = as_task(forked_task)
    
    print("\nForked task structure:")
    print_entity_details(forked_task_typed)
    print("Tags after forking:")
    for tag in forked_task_typed.tags:
        tag_typed = as_tag(tag)
        print_entity_details(tag_typed, prefix="  ")
    
    # Print the lineage
    print_lineage(forked_task_typed)


def test_deep_nesting_forking():
    """Test forking with complex deeply nested structures"""
    print_test_header("Complex Deep Nesting and Circular References")
    
    # Create tags
    tag1 = Tag(name="project", color="#FF0000")
    tag2 = Tag(name="important", color="#0000FF")
    
    # Create address and person
    address = Address(street="123 Main St", city="Anytown", zipcode="12345")
    owner = Person(name="Jane Smith", age=35, address=address)
    member = Person(name="Bob Jones", age=28)
    
    # Create tasks
    task1 = Task(title="Design UI", status="completed", tags=[tag1])
    task2 = Task(title="Implement backend", status="in_progress", tags=[tag2])
    
    # Create a comment
    comment = Comment(content="Looking good!", author=member)
    
    # Create project with everything
    project = Project(
        name="Awesome Project",
        description="A project with complex nested entities",
        tasks=[task1, task2],
        members=[member],
        owner=owner,
        tags=[tag1, tag2],
        comments=[comment]
    )
    
    # Use type assertion for type safety
    project_typed = as_project(project)
    
    print("Original complex structure created.")
    print_entity_details(project_typed)
    
    # Make a deep modification with proper type assertions
    print("\nMaking a deep modification (owner's address)...")
    if project_typed.owner:
        owner_typed = as_person(project_typed.owner)
        if owner_typed.address:
            address_typed = as_address(owner_typed.address)
            address_typed.city = "New City"
    
    # Check for the modification
    stored_address = EntityRegistry.get(address.id)
    print("\nChecking address modifications:")
    if stored_address and project_typed.owner and project_typed.owner.address:
        owner_addr = as_address(project_typed.owner.address)
        addr_modified = track_modifications(stored_address, owner_addr)
    
    # Fork and check the results
    print("\nForking the entire structure...")
    forked_project = project.fork()
    
    # Use type assertion for the forked project
    forked_project_typed = as_project(forked_project)
    
    print("\nVerifying nested references were properly forked:")
    if project_typed.owner:
        print(f"Original owner ID: {project_typed.owner.id}")
    if forked_project_typed.owner:
        print(f"Forked owner ID: {forked_project_typed.owner.id}")
    
    if project_typed.owner and project_typed.owner.address:
        print(f"Original address ID: {project_typed.owner.address.id}")
    if forked_project_typed.owner and forked_project_typed.owner.address:
        print(f"Forked address ID: {forked_project_typed.owner.address.id}")
    
    # Verify that it forked correctly
    print("\nVerifying integrity:")
    if project_typed.id != forked_project_typed.id:
        print("✓ Project has new ID")
    else:
        print("✗ Project has same ID")
        
    if (project_typed.owner and forked_project_typed.owner and 
        project_typed.owner.id != forked_project_typed.owner.id):
        print("✓ Owner has new ID")
    else:
        print("✗ Owner has same ID")
        
    if (project_typed.owner and project_typed.owner.address and 
        forked_project_typed.owner and forked_project_typed.owner.address and
        project_typed.owner.address.id != forked_project_typed.owner.address.id):
        print("✓ Address has new ID")
    else:
        print("✗ Address has same ID")
    
    # Print lineage for the top entity
    print_lineage(forked_project_typed)


def test_efficient_forking():
    """Test that only necessary entities are forked"""
    print_test_header("Efficient Forking (No Unnecessary Forks)")
    
    # Create a nested structure
    tag1 = Tag(name="unchanged", color="#CCCCCC")
    tag2 = Tag(name="changed", color="#DDDDDD")
    address = Address(street="123 Main St", city="Anytown", zipcode="12345")
    person = Person(name="John Doe", age=30, address=address)
    
    # Create project with multiple entities
    project = Project(
        name="Test Project",
        members=[person],
        tags=[tag1, tag2]
    )
    
    # Use type assertion
    project_typed = as_project(project)
    
    print("Original structure created.")
    
    # Modify only one tag - with proper type assertion
    print("\nModifying only one tag...")
    if len(project_typed.tags) > 1:
        tag2_typed = as_tag(project_typed.tags[1])
        tag2_typed.color = "#FF00FF"
    
    # Fork and check what was forked
    print("\nForking the project...")
    forked_project = project.fork()
    
    # Type assertion
    forked_project_typed = as_project(forked_project)
    
    print("\nChecking what was forked:")
    print(f"Original project ID: {project_typed.id}")
    print(f"Forked project ID: {forked_project_typed.id}")
    
    if len(project_typed.members) > 0 and len(forked_project_typed.members) > 0:
        print(f"Original person ID: {project_typed.members[0].id}")
        print(f"Forked person ID: {forked_project_typed.members[0].id}")
    
    if (len(project_typed.members) > 0 and 
        isinstance(project_typed.members[0], Person) and 
        project_typed.members[0].address and
        len(forked_project_typed.members) > 0 and 
        isinstance(forked_project_typed.members[0], Person) and 
        forked_project_typed.members[0].address):
        print(f"Original address ID: {project_typed.members[0].address.id}")
        print(f"Forked address ID: {forked_project_typed.members[0].address.id}")
    
    if len(project_typed.tags) > 0 and len(forked_project_typed.tags) > 0:
        print(f"Original unchanged tag ID: {project_typed.tags[0].id}")
        print(f"Forked unchanged tag ID: {forked_project_typed.tags[0].id}")
    
    if len(project_typed.tags) > 1 and len(forked_project_typed.tags) > 1:
        print(f"Original changed tag ID: {project_typed.tags[1].id}")
        print(f"Forked changed tag ID: {forked_project_typed.tags[1].id}")
    
    # Test if only necessary forks happened
    print("\nVerifying selective forking:")
    if project_typed.id != forked_project_typed.id:
        print("✓ Project was forked (expected)")
    else:
        print("✗ Project was not forked (unexpected)")
        
    if (len(project_typed.members) > 0 and len(forked_project_typed.members) > 0 and
        project_typed.members[0].id == forked_project_typed.members[0].id):
        print("✓ Person was not forked (expected, no changes)")
    else:
        print("✗ Person was forked (unexpected)")
        
    if (len(project_typed.members) > 0 and 
        isinstance(project_typed.members[0], Person) and 
        project_typed.members[0].address and
        len(forked_project_typed.members) > 0 and 
        isinstance(forked_project_typed.members[0], Person) and 
        forked_project_typed.members[0].address and
        project_typed.members[0].address.id == forked_project_typed.members[0].address.id):
        print("✓ Address was not forked (expected, no changes)")
    else:
        print("✗ Address was forked (unexpected)")
        
    if (len(project_typed.tags) > 0 and len(forked_project_typed.tags) > 0 and
        project_typed.tags[0].id == forked_project_typed.tags[0].id):
        print("✓ Unchanged tag was not forked (expected)")
    else:
        print("✗ Unchanged tag was forked (unexpected)")
        
    if (len(project_typed.tags) > 1 and len(forked_project_typed.tags) > 1 and
        project_typed.tags[1].id != forked_project_typed.tags[1].id):
        print("✓ Changed tag was forked (expected)")
    else:
        print("✗ Changed tag was not forked (unexpected)")
    
    # Print lineage for the changed tag
    if len(forked_project_typed.tags) > 1:
        changed_tag = as_tag(forked_project_typed.tags[1])
        print_lineage(changed_tag)


def test_sql_storage():
    """Test SQL storage integration"""
    print_test_header("SQL Storage Integration")
    
    try:
        # Setup SQL storage
        session_factory = setup_sql_storage()
        print("SQL storage set up successfully.")
        
        # Create and register an entity with nested components
        address = Address(street="123 SQL St", city="Database City", zipcode="54321")
        person = Person(name="SQL User", age=25, address=address)
        
        print("Created entities:")
        print_entity_details(person)
        if person.address:
            addr = as_address(person.address)
            print_entity_details(addr, prefix="Address: ")
        
        # Verify they're in the registry
        print("\nVerifying entities are in registry...")
        retrieved_person = EntityRegistry.get(person.id)
        retrieved_address = EntityRegistry.get(address.id)
        
        if retrieved_person and retrieved_address:
            print("✓ Successfully retrieved both entities from registry")
        else:
            print("✗ Failed to retrieve entities from registry")
            
        # Modify and fork
        print("\nModifying and forking...")
        person.name = "Updated SQL User"
        if person.address:
            addr = as_address(person.address)
            addr.city = "New Database City"
        
        forked_person = person.fork()
        forked_person_typed = as_person(forked_person)
        
        print("\nForked entity details:")
        print_entity_details(forked_person_typed)
        if forked_person_typed.address:
            addr = as_address(forked_person_typed.address)
            print_entity_details(addr, prefix="Address: ")
        
        # Verify the forked entities are in the registry
        print("\nVerifying forked entities are in registry...")
        retrieved_forked = EntityRegistry.get(forked_person.id)
        
        if retrieved_forked:
            print("✓ Successfully retrieved forked entity from registry")
            print_entity_details(retrieved_forked)
            
            # Type checking for proper access
            if is_person(retrieved_forked):
                person_retrieved = as_person(retrieved_forked)
                if person_retrieved.address:
                    addr = as_address(person_retrieved.address)
                    print_entity_details(addr, prefix="Address: ")
                    if addr.city == "New Database City":
                        print("✓ Nested address changes were persisted correctly")
                    else:
                        print(f"✗ Nested address changes not persisted. City: {addr.city}")
                else:
                    print("✗ Address not linked to retrieved entity")
            else:
                print("✗ Retrieved entity is not a Person")
        else:
            print("✗ Failed to retrieve forked entity from registry")
        
        # Check lineage tracking
        print("\nChecking lineage tracking...")
        lineage_tree = EntityRegistry.get_lineage_tree_sorted(person.lineage_id)
        print(f"Found {len(lineage_tree['nodes'])} nodes in the lineage tree")
        
        # Print some mermaid diagrams
        print_lineage(forked_person)
        
    except Exception as e:
        print(f"ERROR in SQL storage test: {str(e)}")
        import traceback
        traceback.print_exc()


def test_decorator():
    """Test the entity_tracer decorator"""
    print_test_header("Entity Tracer Decorator")
    
    # Create entities
    entity = SimpleEntity(name="Decorator Test", value=10)
    original_id = entity.id
    
    print("Original entity:")
    print_entity_details(entity)
    
    # Modify using the traced function
    print("\nModifying with decorated function...")
    modified = modify_entity(entity, value=20, description="Added description")
    
    print("\nAfter modification:")
    print_entity_details(modified)
    
    # Check if IDs changed
    print("\nChecking IDs:")
    print(f"Original ID: {original_id}")
    print(f"Current ID: {modified.id}")
    
    if original_id != modified.id:
        print("✓ Entity was automatically forked by the decorator")
    else:
        print("✗ Entity was not forked")
    
    # Check lineage
    print_lineage(modified)


#############################################################################
# Main Test Runner
#############################################################################

def run_all_tests():
    """Run all test cases"""
    print("\n" + "=" * 80)
    print(" ENTITY SYSTEM TESTS ".center(80, "="))
    print("=" * 80 + "\n")
    
    test_simple_entity_forking()
    test_nested_entity_forking()
    test_one_to_many_forking()
    test_deep_nesting_forking()
    test_efficient_forking()
    test_sql_storage()
    test_decorator()
    
    print("\n" + "=" * 80)
    print(" ALL TESTS COMPLETED ".center(80, "="))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_tests()