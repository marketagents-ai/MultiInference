# test_schemas.py
import pytest
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from minference.lite.caregistry import CallableRegistry, validate_schema_compatibility

# Basic Schema Tests
def test_basic_type_schema_derivation(registry):
    """Test schema derivation for basic Python types."""
    def func(x: int, y: float, z: str, b: bool) -> float:
        return float(x)
        
    registry.register("basic_types", func)
    info = registry.get_info("basic_types")
    
    schema = info.input_schema
    assert "x" in schema["properties"]
    assert schema["properties"]["x"]["type"] == "integer"
    assert schema["properties"]["y"]["type"] == "number"
    assert schema["properties"]["z"]["type"] == "string"
    assert schema["properties"]["b"]["type"] == "boolean"

def test_optional_type_schema(registry):
    """Test schema derivation for Optional types."""
    def func(x: int, y: Optional[float] = None) -> float:
        return float(x) + (y or 0.0)
        
    registry.register("optional_types", func)
    info = registry.get_info("optional_types")
    schema = info.input_schema
    
    print("\nFull schema:", schema)  # Debug print
    
    assert "y" in schema["properties"]
    assert schema["required"] == ["x"]  # y is optional
    
    # More flexible assertion for Optional type representation
    y_schema = schema["properties"]["y"]
    is_valid_optional = any([
        # Common representations of Optional types
        isinstance(y_schema.get("type"), list) and set(y_schema["type"]) == {"number", "null"},
        y_schema.get("anyOf") and any(t.get("type") == "null" for t in y_schema["anyOf"]),
        y_schema.get("oneOf") and any(t.get("type") == "null" for t in y_schema["oneOf"]),
        y_schema.get("nullable") is True and y_schema.get("type") == "number"
    ])
    
    assert is_valid_optional, f"Y schema doesn't represent an optional type: {y_schema}"

def test_collection_type_schema(registry):
    """Test schema derivation for collection types."""
    def func(numbers: List[float], metadata: Dict[str, str]) -> float:
        return sum(numbers)
        
    registry.register("collection_types", func)
    info = registry.get_info("collection_types")
    
    schema = info.input_schema
    assert schema["properties"]["numbers"]["type"] == "array"
    assert schema["properties"]["numbers"]["items"]["type"] == "number"
    assert schema["properties"]["metadata"]["type"] == "object"

# Pydantic Model Schema Tests
def test_pydantic_input_schema(registry):
    """Test schema derivation for Pydantic input models."""
    class InputModel(BaseModel):
        name: str = Field(..., description="User name")
        age: int = Field(..., gt=0)
        tags: List[str] = Field(default_factory=list)
        
    def func(data: InputModel) -> dict:
        return data.model_dump()
        
    registry.register("pydantic_input", func)
    info = registry.get_info("pydantic_input")
    
    schema = info.input_schema
    assert schema["properties"]["name"]["description"] == "User name"
    assert "exclusiveMinimum" in schema["properties"]["age"]
    assert schema["properties"]["age"]["exclusiveMinimum"] == 0
    assert schema["properties"]["tags"]["type"] == "array"

def test_pydantic_output_schema(registry):
    """Test schema derivation for Pydantic output models."""
    class OutputModel(BaseModel):
        id: int = Field(..., description="User ID")
        score: float = Field(..., ge=0.0, le=1.0)
        
    def func(x: int) -> OutputModel:
        return OutputModel(id=x, score=0.5)
        
    registry.register("pydantic_output", func)
    info = registry.get_info("pydantic_output")
    
    schema = info.output_schema
    assert "id" in schema["properties"]
    assert schema["properties"]["score"]["minimum"] == 0.0
    assert schema["properties"]["score"]["maximum"] == 1.0

# Schema Compatibility Tests
def test_schema_compatibility_check():
    """Test schema compatibility validation."""
    derived_schema = {
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "number"}
        },
        "required": ["x", "y"]
    }
    
    # Test matching schema
    matching_schema = derived_schema.copy()
    validate_schema_compatibility(derived_schema, matching_schema)
    
    # Test mismatched types
    mismatched_schema = {
        "type": "object",
        "properties": {
            "x": {"type": "string"},  # Wrong type
            "y": {"type": "number"}
        },
        "required": ["x", "y"]
    }
    with pytest.raises(ValueError) as exc:
        validate_schema_compatibility(derived_schema, mismatched_schema)
    assert "type mismatch" in str(exc.value).lower()
    
    # Test missing properties
    missing_prop_schema = {
        "type": "object",
        "properties": {
            "x": {"type": "integer"}
        },
        "required": ["x"]
    }
    with pytest.raises(ValueError) as exc:
        validate_schema_compatibility(derived_schema, missing_prop_schema)
    assert "missing property" in str(exc.value).lower()

# Error Cases
def test_missing_type_hints(registry):
    """Test handling of missing type hints."""
    def no_hints(x):  # Missing type hints
        return x
        
    with pytest.raises(ValueError) as exc:
        registry.register("no_hints", no_hints)
    assert "type hints" in str(exc.value).lower()

def test_invalid_type_hints(registry):
    """Test handling of invalid type hints."""
    def invalid_types(x: "InvalidType") -> int:  #type: ignore
        return 0
        
    with pytest.raises(ValueError) as exc:
        registry.register("invalid_types", invalid_types)
    assert "type hints" in str(exc.value).lower()

def test_complex_nested_schema(registry):
    """Test handling of complex nested schemas."""
    class NestedModel(BaseModel):
        value: float
        
    class ComplexModel(BaseModel):
        name: str
        nested: List[NestedModel]
        metadata: Dict[str, List[int]]
        
    def complex_func(data: ComplexModel) -> ComplexModel:
        return data
        
    registry.register("complex", complex_func)
    info = registry.get_info("complex")
    
    # Verify nested structure
    schema = info.input_schema
    assert schema["properties"]["nested"]["type"] == "array"
    assert "$ref" in schema["properties"]["nested"]["items"]
    ref_name = schema["properties"]["nested"]["items"]["$ref"].split("/")[-1]
    assert "value" in schema["$defs"][ref_name]["properties"]
    assert schema["properties"]["metadata"]["type"] == "object"
    assert schema["properties"]["metadata"]["additionalProperties"]["type"] == "array"

def test_schema_evolution(registry):
    """Test schema updates when function is updated."""
    def original(x: int) -> int:
        return x
        
    def updated(x: int, y: float = 0.0) -> float:
        return float(x) + y
        
    # Register original
    registry.register("evolving", original)
    original_info = registry.get_info("evolving")
    
    # Update function
    registry.update("evolving", updated)
    updated_info = registry.get_info("evolving")
    
    # Verify schema evolution
    assert "y" not in original_info.input_schema["properties"]
    assert "y" in updated_info.input_schema["properties"]
    assert updated_info.input_schema["properties"]["y"]["type"] == "number"