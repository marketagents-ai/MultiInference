"""
Registry implementation for managing immutable Pydantic model instances with lineage-based auto-versioning.

Backwards Compatibility:
- The 'Entity' class still has the same register/save/load methods, just extended with optional
  'lineage_id' and 'parent_id' fields for version tracking.
- The 'EntityRegistry' class still has the same methods (register, get, list_by_type, get_many, etc.),
  but now if the same id is re-registered with changed fields, it *automatically* spawns a new child
  version (mutating the in-memory object). The old ID remains a valid immutable snapshot.
"""

import json
from typing import Dict, Any, Optional, Type, TypeVar, List, Protocol, runtime_checkable, Union, Self
from uuid import UUID, uuid4
from datetime import datetime
from pathlib import Path

import logging
from pydantic import BaseModel, Field, model_validator

from minference.base_registry import BaseRegistry

########################################
# 1) Protocol + Generic
########################################

@runtime_checkable
class HasID(Protocol):
    """Protocol requiring an `id: UUID` field."""
    id: UUID

T = TypeVar('T', bound='Entity')  # We'll define 'Entity' below.


########################################
# 2) The Entity class
########################################

class Entity(BaseModel):
    """
    Base class for registry-integrated, serializable entities.
    
    This class provides:
    1. Integration with EntityRegistry for persistence and retrieval
    2. Basic serialization interface for saving/loading
    3. Common entity attributes and operations
    
    Subclasses are responsible for:
    1. Implementing custom serialization if needed (_custom_serialize/_custom_deserialize)
    2. Handling any nested entities or complex relationships
    3. Managing entity-specific validation and business logic
    
    All entities are conceptually immutable: modifications require new versions (UUIDs).
    However, in this auto-versioning design, if you re-register the same object with changed fields,
    the registry will automatically spawn a new version in-place (new id, updated parent_id, etc.).
    """

    # Original fields
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this entity instance (version)."
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when this entity was created."
    )

    # New optional fields for lineage-based versioning
    lineage_id: UUID = Field(
        default_factory=uuid4,
        description="Stable ID representing the entire lineage/family of versions."
    )
    parent_id: Optional[UUID] = Field(
        default=None,
        description="If set, points to the immediate parent's version ID in the lineage."
    )

    class Config:
        # Add JSON encoders for UUID and datetime
        json_encoders = {
            UUID: str,  # Convert UUID to string
            datetime: lambda v: v.isoformat()  # Convert datetime to ISO format string
        }

    @model_validator(mode='after')
    def register_entity(self) -> Self:
        """Register this entity instance in the registry (auto-calls EntityRegistry.register)."""
        from __main__ import EntityRegistry  # or from your_module import EntityRegistry
        # If you're inside a package, adjust the import so it references the correct module path.

        registry = EntityRegistry
        registry._logger.info(f"{self.__class__.__name__}({self.id}): Registering entity")

        try:
            registry.register(self)
            registry._logger.info(f"{self.__class__.__name__}({self.id}): Successfully registered")
        except Exception as e:
            registry._logger.error(f"{self.__class__.__name__}({self.id}): Registration failed - {str(e)}")
            raise ValueError(f"Entity registration failed: {str(e)}") from e

        return self

    def _custom_serialize(self) -> Dict[str, Any]:
        """
        Custom serialization hook for subclasses.
        
        Override this method to add custom serialization logic.
        The result will be included in the serialized output under 'custom_data'.
        """
        return {}

    @classmethod
    def _custom_deserialize(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Custom deserialization hook for subclasses.
        
        Override this method to handle custom deserialization logic.
        The input comes from the 'custom_data' field in serialized data.
        
        Args:
            data: Custom data from serialized entity
            
        Returns:
            Dict of deserialized fields to include in entity initialization
        """
        return {}

    def save(self, path: Path) -> None:
        """
        Save this entity instance to a file.

        Args:
            path: Path where to save the entity

        Raises:
            IOError: If saving fails
        """
        from __main__ import EntityRegistry

        registry = EntityRegistry
        registry._logger.info(f"{self.__class__.__name__}({self.id}): Saving to {path}")

        try:
            # Get base serialization
            data = self.model_dump()

            # Add metadata
            metadata = {
                "entity_type": self.__class__.__name__,
                "schema_version": "1.0",
                "saved_at": datetime.utcnow().isoformat()
            }

            # Get custom data
            custom_data = self._custom_serialize()

            # Combine all
            serialized = {
                "metadata": metadata,
                "data": data,
                "custom_data": custom_data
            }

            # Save
            with open(path, 'w') as f:
                json.dump(serialized, f, indent=2)

            registry._logger.info(f"{self.__class__.__name__}({self.id}): Successfully saved")

        except Exception as e:
            registry._logger.error(f"{self.__class__.__name__}({self.id}): Save failed - {str(e)}")
            raise IOError(f"Failed to save entity: {str(e)}") from e

    @classmethod
    def load(cls: Type[T], path: Path) -> T:
        """
        Load an entity instance from a file.

        Args:
            path: Path to the saved entity

        Returns:
            Loaded entity instance

        Raises:
            IOError: If loading fails
            ValueError: If data validation fails
        """
        from __main__ import EntityRegistry

        registry = EntityRegistry
        registry._logger.info(f"{cls.__name__}: Loading from {path}")

        try:
            # Load file
            with open(path) as f:
                serialized = json.load(f)

            # Verify entity type
            metadata = serialized["metadata"]
            if metadata["entity_type"] != cls.__name__:
                raise ValueError(
                    f"Entity type mismatch. File contains {metadata['entity_type']}, "
                    f"expected {cls.__name__}"
                )

            # Get base and custom data
            base_data = serialized["data"]
            custom_data = cls._custom_deserialize(serialized.get("custom_data", {}))

            # Create instance
            instance = cls(**{**base_data, **custom_data})
            registry._logger.info(f"{cls.__name__}({instance.id}): Successfully loaded")
            return instance

        except Exception as e:
            registry._logger.error(f"{cls.__name__}: Load failed - {str(e)}")
            raise IOError(f"Failed to load entity: {str(e)}") from e

    @classmethod
    def get(cls: Type[T], entity_id: UUID) -> Optional[T]:
        """Get an entity instance from the registry."""
        from __main__ import EntityRegistry
        return EntityRegistry.get(entity_id, expected_type=cls)

    @classmethod
    def list_all(cls: Type[T]) -> List[T]:
        """List all entities of this type."""
        from __main__ import EntityRegistry
        return EntityRegistry.list_by_type(cls)

    @classmethod
    def get_many(cls: Type[T], entity_ids: List[UUID]) -> List[T]:
        """Get multiple entities by their IDs."""
        from __main__ import EntityRegistry
        return EntityRegistry.get_many(entity_ids, expected_type=cls)


########################################
# 3) The EntityRegistry class
########################################

EntityType = TypeVar('EntityType', bound=HasID)

class EntityRegistry(BaseRegistry[EntityType]):
    """
    Registry for managing immutable Pydantic model instances with lineage-based versioning.
    
    *Backwards Compatibility:*
    - We maintain the same method signatures (register, get, list_by_type, etc.).
    - If the same ID is re-registered with changed fields, we now auto-create a new version
      by mutating the entity's id + parent_id + created_at, leaving the old version under the old ID.
    """
    _registry: Dict[UUID, EntityType] = {}
    _timestamps: Dict[UUID, datetime] = {}
    _inference_orchestrator: Optional[object] = None

    # optional lineage map: lineage_id -> list of version IDs
    _lineages: Dict[UUID, List[UUID]] = {}

    @classmethod
    def register(cls, entity: EntityType) -> None:
        """
        Register a new entity instance or verify reference to existing entity.

        If the entity.id is already in the registry but the "payload fields" differ,
        we transform 'entity' in-place into a new version:
          - entity.parent_id = old_id (if 'parent_id' is a field)
          - entity.id = new random UUID
          - entity.created_at = datetime.utcnow() (if 'created_at' is a field)
        The old version remains in _registry under the old ID.

        If the content is identical, do nothing (already registered).
        """
        if not isinstance(entity, BaseModel):
            cls._logger.error(f"Invalid entity type: {type(entity)}")
            raise ValueError("Entity must be a Pydantic model instance")
        if not isinstance(entity, HasID):
            cls._logger.error(f"Entity missing ID field: {type(entity)}")
            raise ValueError("Entity must have an 'id' field")

        entity_id = entity.id
        cls._logger.debug(f"Attempting to register {entity.__class__.__name__}({entity_id})")

        if entity_id in cls._registry:
            existing_entity = cls._registry[entity_id]

            # Compare type
            if type(existing_entity) != type(entity):
                cls._logger.error(
                    f"Type mismatch for {entity_id}:\n"
                    f"Existing: {type(existing_entity)}\n"
                    f"New: {type(entity)}"
                )
                raise ValueError(f"Entity type mismatch for {entity_id}")

            # Exclude IDs/lineage fields from the mismatch check
            exclude_keys = {'id', 'created_at'}
            if hasattr(entity, 'lineage_id'):
                exclude_keys.add('lineage_id')
            if hasattr(entity, 'parent_id'):
                exclude_keys.add('parent_id')

            old_data = existing_entity.model_dump(exclude=exclude_keys)
            new_data = entity.model_dump(exclude=exclude_keys)

            if old_data == new_data:
                # They match => do nothing
                cls._logger.debug(
                    f"{entity.__class__.__name__}({entity_id}) already registered and matches"
                )
                return
            else:
                # Auto-create new version
                cls._logger.info(
                    f"Detected changes in re-registered {entity.__class__.__name__}({entity_id}); "
                    "creating a new version in place."
                )
                # set parent_id if present
                if hasattr(entity, 'parent_id'):
                    setattr(entity, 'parent_id', entity_id)
                # new ID
                new_id = uuid4()
                setattr(entity, 'id', new_id)
                # reset created_at if present
                if hasattr(entity, 'created_at'):
                    setattr(entity, 'created_at', datetime.utcnow())

                entity_id = new_id

        # Normal registration logic
        cls._registry[entity_id] = entity
        cls._timestamps[entity_id] = datetime.utcnow()
        cls._logger.info(f"Successfully registered {entity.__class__.__name__}({entity_id})")

        # If there's a lineage_id, track it
        if hasattr(entity, 'lineage_id'):
            lineage_val = getattr(entity, 'lineage_id')
            if isinstance(lineage_val, UUID):
                if lineage_val not in cls._lineages:
                    cls._lineages[lineage_val] = []
                if entity_id not in cls._lineages[lineage_val]:
                    cls._lineages[lineage_val].append(entity_id)

    @classmethod
    def get(
        cls,
        entity_id: UUID,
        expected_type: Optional[Type[EntityType]] = None
    ) -> Optional[EntityType]:
        """
        Retrieve an immutable entity by ID with optional type checking.
        Backwards-compatible signature.
        """
        cls._logger.debug(f"Retrieving entity {entity_id}")
        entity = cls._registry.get(entity_id)
        if entity is None:
            cls._logger.debug(f"Entity {entity_id} not found")
            return None
        if expected_type and not isinstance(entity, expected_type):
            cls._logger.error(
                f"Type mismatch for {entity_id}. "
                f"Expected {expected_type.__name__}, got {type(entity).__name__}"
            )
            return None
        return entity

    @classmethod
    def list_by_type(cls, entity_type: Type[EntityType]) -> List[EntityType]:
        """List all entities of a specific type (unchanged)."""
        cls._logger.debug(f"Listing entities of type {entity_type.__name__}")
        return [
            e for e in cls._registry.values()
            if isinstance(e, entity_type)
        ]

    @classmethod
    def get_many(
        cls,
        entity_ids: List[UUID],
        expected_type: Optional[Type[EntityType]] = None
    ) -> List[EntityType]:
        """Get multiple entities by their IDs (unchanged)."""
        cls._logger.debug(f"Retrieving {len(entity_ids)} entities")
        return [
            e for eid in entity_ids
            if (e := cls.get(eid, expected_type=expected_type)) is not None
        ]

    @classmethod
    def get_registry_status(cls) -> Dict[str, Any]:
        """Get detailed status of the registry, including lineage info."""
        base_status = super().get_registry_status()

        # entity-specific stats
        type_counts: Dict[str, int] = {}
        for e in cls._registry.values():
            nm = e.__class__.__name__
            type_counts[nm] = type_counts.get(nm, 0) + 1

        # timestamp-based versioning
        timestamps = sorted(cls._timestamps.values())

        total_lineages = len(cls._lineages)
        total_versions = sum(len(v) for v in cls._lineages.values())

        return {
            **base_status,
            "entities_by_type": type_counts,
            "version_history": {
                "first_version": timestamps[0].isoformat() if timestamps else None,
                "latest_version": timestamps[-1].isoformat() if timestamps else None,
                "version_count": len(timestamps)
            },
            "total_lineages": total_lineages,
            "total_versions": total_versions,
        }

    @classmethod
    def set_inference_orchestrator(cls, inference_orchestrator: object) -> None:
        """Set the inference orchestrator (unchanged)."""
        cls._inference_orchestrator = inference_orchestrator

    @classmethod
    def get_inference_orchestrator(cls) -> Optional[object]:
        """Get the inference orchestrator (unchanged)."""
        return cls._inference_orchestrator

    # Additional helpers for lineage
    @classmethod
    def get_lineage_ids(cls, lineage_id: UUID) -> List[UUID]:
        """Return the list of version IDs in this lineage."""
        return cls._lineages.get(lineage_id, [])

    @classmethod
    def get_lineage_tree_sorted(cls, lineage_id: UUID) -> List[EntityType]:
        """
        Return all entities in a lineage, sorted by created_at ascending.
        """
        version_ids = cls._lineages.get(lineage_id, [])
        # filter out missing from _registry
        entities = [cls._registry[vid] for vid in version_ids if vid in cls._registry]
        entities.sort(key=lambda e: getattr(e, 'created_at', datetime.min))
        return entities


########################################
# 4) OPTIONAL DEMO
########################################

if __name__ == "__main__":
    # This is a small example to show how it works
    from pydantic import BaseModel, Field
    registry = EntityRegistry()
    # Clear everything
    EntityRegistry.clear()
    EntityRegistry.clear_logs()

    # Subclass for testing
    class MyEntity(Entity):
        some_data: str = "initial"

    # 1) Create root entity
    root = MyEntity(some_data="Hello")
    # Because of the model_validator, it auto-calls register.

    # 2) Re-register with no changes => no new version
    EntityRegistry.register(root)

    # 3) Now let's change 'root.some_data'
    root.some_data = "New stuff"
    # On re-register, we auto-create a new version in place
    EntityRegistry.register(root)

    print("Root after re-register =>", root)
    # The 'root.id' is now a new UUID; old version remains in registry

    # 4) Show lineage
    lineage_list = EntityRegistry.get_lineage_ids(root.lineage_id)
    print("Lineage =>", lineage_list)
    sorted_entities = EntityRegistry.get_lineage_tree_sorted(root.lineage_id)
    print("Lineage sorted by created_at =>")
    for ent in sorted_entities:
        print("   =>", ent.id, ent.some_data, ent.created_at.isoformat())

    # Logs
    print("\nLOGS:\n", EntityRegistry.get_logs())
