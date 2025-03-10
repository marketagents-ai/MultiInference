"""
Inspect the Thread models to understand their fields and structures.
"""

from pprint import pprint
import inspect
from minference.threads.models import (
    ChatThread, ChatMessage, SystemPrompt, LLMConfig, 
    CallableTool, StructuredTool, Usage, GeneratedJsonObject,
    RawOutput, ProcessedOutput
)

def print_model_fields(model_class):
    """Print fields of a model class with their types."""
    print(f"\n{'='*40}")
    print(f"{model_class.__name__} Fields:")
    print(f"{'='*40}")
    
    # Get the model's fields from its __annotations__
    if hasattr(model_class, "__annotations__"):
        for field_name, field_type in model_class.__annotations__.items():
            print(f"{field_name}: {field_type}")
    
    # Get the model's __init__ parameters
    print(f"\n{model_class.__name__} __init__ parameters:")
    sig = inspect.signature(model_class.__init__)
    for name, param in sig.parameters.items():
        if name != 'self':
            print(f"{name}: {param.annotation}")
            
    # Print default values if available
    print(f"\n{model_class.__name__} default values:")
    for name, param in sig.parameters.items():
        if name != 'self' and param.default is not param.empty:
            print(f"{name}: {param.default}")
    
    # Print model's direct fields from the model
    if hasattr(model_class, "model_fields"):
        print(f"\n{model_class.__name__} model_fields:")
        for field_name, field_info in model_class.model_fields.items():
            required = not field_info.is_optional()
            print(f"{field_name}: required={required}, default={field_info.get_default()}")

# Inspect single model for detailed analysis
usage_model = Usage
print_model_fields(usage_model)