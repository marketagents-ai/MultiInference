"""
Example demonstrating sequential tool execution with structured outputs.
This example shows how to:
1. Register multiple tools with structured outputs
2. Use tools in sequence
3. Handle structured data between tool calls
"""
import asyncio
import logging
from minference.lite.inference import InferenceOrchestrator
from minference.lite.models import ChatThread, LLMClient, LLMConfig, MessageRole, CallableTool, ResponseFormat
from minference.caregistry import CallableRegistry
from minference.enregistry import EntityRegistry
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO)
EntityRegistry._logger = logging.getLogger(__name__)
CallableRegistry._logger = logging.getLogger(__name__)

# Initialize registries
EntityRegistry()
CallableRegistry()

# Register tools for a simple data processing pipeline
def fetch_stock_data(symbol: str) -> Dict[str, Any]:
    """Fetch stock data (mock implementation)."""
    # Mock data
    return {
        "symbol": symbol,
        "price": 150.25,
        "volume": 1000000,
        "change": 2.5
    }

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment of text (mock implementation)."""
    return {
        "text": text,
        "sentiment": "positive",
        "confidence": 0.85
    }

def generate_trade_recommendation(
    stock_data: Dict[str, Any],
    sentiment: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate trade recommendation based on data and sentiment."""
    return {
        "symbol": stock_data["symbol"],
        "action": "BUY" if sentiment["sentiment"] == "positive" else "SELL",
        "confidence": sentiment["confidence"] * (stock_data["change"] / 5),
        "target_price": stock_data["price"] * 1.1
    }

# Register the tools
CallableRegistry.register("fetch_stock_data", fetch_stock_data)
CallableRegistry.register("analyze_sentiment", analyze_sentiment)
CallableRegistry.register("generate_trade_recommendation", generate_trade_recommendation)

# Create CallableTools
fetch_stock_tool = CallableTool.from_callable(fetch_stock_data)
analyze_sentiment_tool = CallableTool.from_callable(analyze_sentiment)
generate_recommendation_tool = CallableTool.from_callable(generate_trade_recommendation)

async def main():
    # Initialize the orchestrator
    orchestrator = InferenceOrchestrator()

    # Create a chat thread with sequential tool usage
    chat_thread = ChatThread(
        llm_config=LLMConfig(
            client=LLMClient.anthropic,
            model="claude-3-opus-20240229",
            response_format=ResponseFormat.json_beg,
            max_tokens=1000,
            temperature=0
        ),
        tools=[fetch_stock_tool, analyze_sentiment_tool, generate_recommendation_tool]
    )
    
    # Add the message with clear instructions
    message = """Please help me analyze AAPL stock by following these steps:
    1. Use the fetch_stock_data tool to get current AAPL stock data
    2. Use the analyze_sentiment tool to analyze this news: 'Apple reports record iPhone sales'
    3. Use the generate_trade_recommendation tool with the data from steps 1 and 2 to get a recommendation
    
    Please execute these tools in sequence and show me the results of each step.
    Please provide your response in JSON format.
    """
    chat_thread.new_message = message
    chat_thread.add_user_message()

    try:
        # Run inference
        results = await orchestrator.run_parallel_ai_completion([chat_thread])
        result = results[0]  # Get the first result since we only have one thread
        
        # Print results
        print("\nAnalysis Results:")
        print(f"Response: {result.content}")
        if result.json_object:
            print("\nTool Call:")
            print(f"Name: {result.json_object.name}")
            print(f"Arguments: {result.json_object.object}")

    except Exception as e:
        print(f"Error during inference: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 