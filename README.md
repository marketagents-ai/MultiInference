# MarketInference

A multi-agent parallel inference orchestrator for managing and coordinating AI model interactions. This library provides a robust framework for handling parallel AI completions, tool management, and structured outputs across multiple AI providers.

## Features

### Multi-Provider Support
- OpenAI (GPT models)
- Anthropic (Claude models)
- VLLM
- LiteLLM

### Core Capabilities
- Parallel request processing with intelligent rate limiting
- Tool registration and execution framework
- Structured message handling and history tracking
- Entity management with immutable patterns
- Comprehensive logging and monitoring
- Type-safe implementations using Pydantic

### Advanced Features
- Automatic schema validation for tools and outputs
- Asynchronous and synchronous execution support
- UUID-based entity tracking
- Configurable rate limiting per provider
- Message history management
- Tool execution orchestration

## Dependencies

Key dependencies and their purposes:
- **AI/ML**:
  - `anthropic`: Anthropic Claude API integration
  - `openai`: OpenAI API integration
  - `tiktoken`: Token counting for rate limiting
- **Web/API**:
  - `fastapi`: API framework support
  - `uvicorn`: ASGI server
  - `aiohttp`: Async HTTP client
- **Data Handling**:
  - `pydantic`: Data validation and settings management
  - `sqlmodel`: SQL database integration
  - `polars`: Data manipulation
- **Development**:
  - `pytest`: Testing framework
  - `pytest-asyncio`: Async test support
  - `pytest-cov`: Test coverage
  - `pytest-benchmark`: Performance testing

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MarketInference.git
cd MarketInference

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

## Environment Setup

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Configure the following environment variables in `.env`:
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_key
OPENAI_CONTEXT_LENGTH=8192

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_key
ANTHROPIC_CONTEXT_LENGTH=100000

# Azure OpenAI (Optional)
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_CONTEXT_LENGTH=8192

# Rate Limiting
MAX_REQUESTS_PER_MINUTE=50
MAX_TOKENS_PER_MINUTE=100000
```

## Quick Start

1. Basic usage example:
```python
from minference.lite import InferenceOrchestrator
from minference.lite.models import ChatThread, LLMClient

# Initialize the orchestrator
orchestrator = InferenceOrchestrator()

# Create a chat thread
chat_thread = ChatThread(
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    llm_client=LLMClient(provider="openai", model="gpt-4")
)

# Run inference
result = await orchestrator.run_inference(chat_thread)
print(result.content)
```

2. Using tools:
```python
from minference.caregistry import CallableRegistry

# Register a tool
@CallableRegistry.register
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b

# Use in chat with tool calling
chat_thread = ChatThread(
    messages=[{"role": "user", "content": "What is 5 + 3?"}],
    llm_client=LLMClient(
        provider="anthropic",
        model="claude-3-opus-20240229",
        tools=[calculate_sum]
    )
)
```

## Example Files

The `examples/` directory contains several example implementations:

1. `lite_inference_example.py`: Basic inference usage
```python
# Example of basic inference with a single model
from minference.lite import InferenceOrchestrator
# ... (see examples/lite_inference_example.py)
```

2. `lite_sequential_tool_inference.py`: Sequential tool execution
```python
# Example of sequential tool execution
from minference.lite.models import ChatThread
# ... (see examples/lite_sequential_tool_inference.py)
```

3. `lite_tools_inference_example.py`: Tool integration examples
```python
# Example of tool integration
from minference.caregistry import CallableRegistry
# ... (see examples/lite_tools_inference_example.py)
```

4. `tools_setup_example.py`: Complex tool setup
```python
# Example of advanced tool configuration
from minference.lite.models import StructuredTool
# ... (see examples/tools_setup_example.py)
```

5. `pathfinder.ipynb`: Interactive notebook with examples

## Advanced Usage

### Parallel Processing

```python
# Process multiple chat threads in parallel
chat_threads = [
    ChatThread(...),
    ChatThread(...),
    ChatThread(...)
]

results = await orchestrator.run_parallel_inference(chat_threads)
```

### Custom Tool Registration

```python
from minference.caregistry import CallableRegistry
from typing import Dict, Any

@CallableRegistry.register
async def fetch_data(url: str) -> Dict[str, Any]:
    """Fetch data from an API endpoint."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

### Entity Management

```python
from minference.lite.models import Entity
from uuid import UUID

class CustomEntity(Entity):
    name: str
    data: Dict[str, Any]
    
    def process(self) -> None:
        # Process entity data
        pass
```

## Configuration

The library can be configured through environment variables or programmatically:

```python
from minference.lite.models import RequestLimits

# Configure rate limits
limits = RequestLimits(
    max_requests_per_minute=50,
    max_tokens_per_minute=100000,
    provider="openai"
)
```

## Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run with coverage
pytest --cov=minference tests/
```

## Project Structure

```
MarketInference/
├── minference/
│   ├── lite/               # Lightweight implementation
│   │   ├── inference.py    # Core inference logic
│   │   ├── models.py       # Data models
│   │   └── requests.py     # Request handling
│   ├── core/               # Core functionality
│   │   └── clients_models.py # Client implementations
│   ├── base_registry.py    # Base registry implementation
│   ├── caregistry.py      # Callable registry
│   ├── enregistry.py      # Entity registry
│   └── utils.py           # Utility functions
├── tests/                  # Test suite
│   ├── test_fixtures/     # Test fixtures
│   ├── test_registration.py
│   ├── test_schemas.py
│   └── test_execution.py
├── examples/               # Usage examples
└── requirements.txt        # Dependencies
```

