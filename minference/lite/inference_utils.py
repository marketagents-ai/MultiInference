"""
Direct extraction of InferenceOrchestrator methods as standalone functions.
Maintains exact behavior while making functions reusable outside the orchestrator.
"""
import os
import json
import time
from typing import Dict, Any, Optional, List
from pydantic import ValidationError
from uuid import UUID
from anthropic.types.message_create_params import ToolChoiceToolChoiceTool, ToolChoiceToolChoiceAuto
from minference.lite.models import ChatThread, LLMClient
from minference.clients_models import AnthropicRequest, OpenAIRequest, VLLMRequest
from minference.lite.oai_parallel import OAIApiFromFileConfig
from minference.enregistry import EntityRegistry

def prepare_requests_file(chat_threads: List[ChatThread], client: str, filename: str):
    """Prepare JSONL file with chat thread requests."""
    requests = []
    EntityRegistry._logger.info(f"Preparing {client} requests for {len(chat_threads)} chat threads")
    for chat_thread in chat_threads:
        request = convert_chat_thread_to_request(chat_thread, client)
        if request:
            metadata = {
                "chat_thread_id": str(chat_thread.id),
                "start_time": time.time(),
                "end_time": None,
                "total_time": None
            }
            requests.append([metadata, request])
    
    with open(filename, 'w') as f:
        for request in requests:
            json.dump(request, f)
            f.write('\n')
    EntityRegistry._logger.debug(f"Wrote {len(requests)} requests to {filename}")

def validate_anthropic_request(request: Dict[str, Any]) -> bool:
    """Validate an Anthropic API request."""
    try:
        anthropic_request = AnthropicRequest(**request)
        return True
    except Exception as e:
        EntityRegistry._logger.error(f"Error validating Anthropic request: {e}")
        raise ValidationError(f"Error validating Anthropic request: {e} with request: {request}")

def validate_openai_request(request: Dict[str, Any]) -> bool:
    """Validate an OpenAI API request."""
    try:
        openai_request = OpenAIRequest(**request)
        return True
    except Exception as e:
        EntityRegistry._logger.error(f"Error validating OpenAI request: {e}")
        raise ValidationError(f"Error validating OpenAI request: {e} with request: {request}")

def validate_vllm_request(request: Dict[str, Any]) -> bool:
    """Validate a vLLM API request."""
    try:
        vllm_request = VLLMRequest(**request)
        return True
    except Exception as e:
        EntityRegistry._logger.error(f"Error validating VLLM request: {e}")
        raise ValidationError(f"Error validating VLLM request: {e} with request: {request}")

def get_openai_request(chat_thread: ChatThread) -> Optional[Dict[str, Any]]:
    """Get OpenAI format request from chat thread."""
    messages = chat_thread.oai_messages
    request = {
        "model": chat_thread.llm_config.model,
        "messages": messages,
        "max_tokens": chat_thread.llm_config.max_tokens,
        "temperature": chat_thread.llm_config.temperature,
    }
    if chat_thread.oai_response_format:
        request["response_format"] = chat_thread.oai_response_format
    if chat_thread.llm_config.response_format == "tool" and chat_thread.forced_output:
        tool = chat_thread.forced_output
        if tool:
            request["tools"] = [tool.get_openai_tool()]
            request["tool_choice"] = {"type": "function", "function": {"name": tool.name}}
    elif chat_thread.llm_config.response_format == "auto_tools":
        tools = chat_thread.tools
        if tools:
            request["tools"] = [t.get_openai_tool() for t in tools]
            request["tool_choice"] = "auto"
    if validate_openai_request(request):
        return request
    else:
        return None

def get_anthropic_request(chat_thread: ChatThread) -> Optional[Dict[str, Any]]:
    """Get Anthropic format request from chat thread."""
    system_content, messages = chat_thread.anthropic_messages    
    request = {
        "model": chat_thread.llm_config.model,
        "max_tokens": chat_thread.llm_config.max_tokens,
        "temperature": chat_thread.llm_config.temperature,
        "messages": messages,
        "system": system_content if system_content else None
    }
    if chat_thread.llm_config.response_format == "tool" and chat_thread.forced_output:
        tool = chat_thread.forced_output
        if tool:
            request["tools"] = [tool.get_anthropic_tool()]
            request["tool_choice"] = ToolChoiceToolChoiceTool(name=tool.name, type="tool")
    elif chat_thread.llm_config.response_format == "auto_tools":
        tools = chat_thread.tools
        if tools:
            request["tools"] = [t.get_anthropic_tool() for t in tools]
            request["tool_choice"] = ToolChoiceToolChoiceAuto(type="auto")

    if validate_anthropic_request(request):
        return request
    else:
        return None

def get_vllm_request(chat_thread: ChatThread) -> Optional[Dict[str, Any]]:
    """Get vLLM format request from chat thread."""
    messages = chat_thread.vllm_messages
    request = {
        "model": chat_thread.llm_config.model,
        "messages": messages,
        "max_tokens": chat_thread.llm_config.max_tokens,
        "temperature": chat_thread.llm_config.temperature,
    }
    if chat_thread.llm_config.response_format == "tool" and chat_thread.forced_output:
        tool = chat_thread.forced_output
        if tool:
            request["tools"] = [tool.get_openai_tool()]
            request["tool_choice"] = {"type": "function", "function": {"name": tool.name}}
    if chat_thread.llm_config.response_format == "json_object":
        raise ValueError("VLLM does not support json_object response format otherwise infinite whitespaces are returned")
    if chat_thread.oai_response_format and chat_thread.oai_response_format:
        request["response_format"] = chat_thread.oai_response_format
    
    if validate_vllm_request(request):
        return request
    else:
        return None

def get_litellm_request(chat_thread: ChatThread) -> Optional[Dict[str, Any]]:
    """Get LiteLLM format request from chat thread."""
    if chat_thread.llm_config.response_format == "json_object":
        raise ValueError("VLLM does not support json_object response format otherwise infinite whitespaces are returned")
    return get_openai_request(chat_thread)

def convert_chat_thread_to_request(chat_thread: ChatThread, client: str) -> Optional[Dict[str, Any]]:
    """Convert chat thread to client-specific request format."""
    if client == "openai":
        return get_openai_request(chat_thread)
    elif client == "anthropic":
        return get_anthropic_request(chat_thread)
    elif client == "vllm":
        return get_vllm_request(chat_thread)
    elif client == "litellm":
        return get_litellm_request(chat_thread)
    else:
        raise ValueError(f"Invalid client: {client}")

def create_oai_completion_config(
    chat_thread: ChatThread, 
    requests_file: str, 
    results_file: str,
    openai_key: str,
    max_requests_per_minute: int,
    max_tokens_per_minute: int
) -> Optional[OAIApiFromFileConfig]:
    """Create OpenAI completion configuration."""
    if chat_thread.llm_config.client == "openai" and openai_key:
        return OAIApiFromFileConfig(
            requests_filepath=requests_file,
            save_filepath=results_file,
            request_url="https://api.openai.com/v1/chat/completions",
            api_key=openai_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name="cl100k_base",
            max_attempts=5,
            logging_level=20,
        )
    return None

def create_anthropic_completion_config(
    chat_thread: ChatThread, 
    requests_file: str, 
    results_file: str,
    anthropic_key: str,
    max_requests_per_minute: int,
    max_tokens_per_minute: int
) -> Optional[OAIApiFromFileConfig]:
    """Create Anthropic completion configuration."""
    if chat_thread.llm_config.client == "anthropic" and anthropic_key:
        return OAIApiFromFileConfig(
            requests_filepath=requests_file,
            save_filepath=results_file,
            request_url="https://api.anthropic.com/v1/messages",
            api_key=anthropic_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name="cl100k_base",
            max_attempts=5,
            logging_level=20,
        )
    return None

def create_vllm_completion_config(
    chat_thread: ChatThread, 
    requests_file: str, 
    results_file: str,
    vllm_endpoint: str,
    vllm_key: Optional[str],
    max_requests_per_minute: int,
    max_tokens_per_minute: int
) -> Optional[OAIApiFromFileConfig]:
    """Create vLLM completion configuration."""
    if chat_thread.llm_config.client == "vllm":
        return OAIApiFromFileConfig(
            requests_filepath=requests_file,
            save_filepath=results_file,
            request_url=vllm_endpoint,
            api_key=vllm_key if vllm_key else "",
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name="cl100k_base",
            max_attempts=5,
            logging_level=20,
        )
    return None

def create_litellm_completion_config(
    chat_thread: ChatThread, 
    requests_file: str, 
    results_file: str,
    litellm_endpoint: str,
    litellm_key: Optional[str],
    max_requests_per_minute: int,
    max_tokens_per_minute: int
) -> Optional[OAIApiFromFileConfig]:
    """Create LiteLLM completion configuration."""
    if chat_thread.llm_config.client == "litellm":
        return OAIApiFromFileConfig(
            requests_filepath=requests_file,
            save_filepath=results_file,
            request_url=litellm_endpoint,
            api_key=litellm_key if litellm_key else "",
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name="cl100k_base",
            max_attempts=5,
            logging_level=20,
        )
    return None