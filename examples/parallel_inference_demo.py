"""
Example demonstrating parallel inference with multiple tools.
"""
from typing import Dict, Any
import asyncio
import random
import logging
from datetime import datetime

from minference.lite.inference import InferenceOrchestrator
from minference.lite.models import LLMConfig, LLMClient, ResponseFormat, ChatThread, MessageRole, CallableTool
from minference.caregistry import CallableRegistry
from minference.enregistry import EntityRegistry

# Set up logging
logging.basicConfig(level=logging.INFO)
EntityRegistry._logger = logging.getLogger(__name__)
CallableRegistry._logger = logging.getLogger(__name__)

# Initialize registries
EntityRegistry()
CallableRegistry()

def get_current_weather(location: str, unit: str = "fahrenheit") -> Dict[str, Any]:
    """Get the current weather in a given location."""
    # Simulate API call
    temperature = random.randint(0, 100)
    return {
        "location": location,
        "temperature": temperature,
        "unit": unit,
        "timestamp": datetime.now().isoformat()
    }

def get_stock_price(symbol: str) -> Dict[str, Any]:
    """Get the current stock price for a given symbol."""
    # Simulate API call
    price = random.uniform(10, 1000)
    return {
        "symbol": symbol,
        "price": round(price, 2),
        "currency": "USD",
        "timestamp": datetime.now().isoformat()
    }

# Register tools
CallableRegistry.register("get_current_weather", get_current_weather)
CallableRegistry.register("get_stock_price", get_stock_price)

# Create CallableTools
weather_tool = CallableTool.from_callable(get_current_weather)
stock_tool = CallableTool.from_callable(get_stock_price)

# Create orchestrator
orchestrator = InferenceOrchestrator()

# Run parallel inference
async def main():
    chat_threads = [
        ChatThread(
            llm_config=LLMConfig(
                client=LLMClient.openai,
                model="gpt-4",
                response_format=ResponseFormat.tool,  # Changed back to tool for OpenAI
                max_tokens=1000,
                temperature=0
            ),
            tools=[weather_tool]
        ),
        ChatThread(
            llm_config=LLMConfig(
                client=LLMClient.openai,
                model="gpt-4",
                response_format=ResponseFormat.tool,  # Changed back to tool for OpenAI
                max_tokens=1000,
                temperature=0
            ),
            tools=[stock_tool]
        )
    ]
    
    # Add messages to threads with clear instructions
    weather_message = """Please use the get_current_weather tool to check the weather in New York and Tokyo. 
    I need actual temperature data for both cities. Make sure to call the tool twice, once for each city.
    Please provide your response in a clear, structured format."""
    chat_threads[0].new_message = weather_message
    chat_threads[0].add_user_message()
    
    stock_message = """Please use the get_stock_price tool to check the current stock prices for AAPL and GOOGL. 
    I need the actual current prices for both stocks. Make sure to call the tool twice, once for each stock symbol.
    Please provide your response in a clear, structured format."""
    chat_threads[1].new_message = stock_message
    chat_threads[1].add_user_message()
    
    results = await orchestrator.run_parallel_ai_completion(chat_threads)
    for i, result in enumerate(results):
        print(f"\nQuery: {chat_threads[i].history[-1].content}")
        print(f"Response: {result.content}")
        if result.json_object:
            print(f"Tool Call: {result.json_object.name}")
            print(f"Arguments: {result.json_object.object}")
        print("---")

if __name__ == "__main__":
    asyncio.run(main()) 