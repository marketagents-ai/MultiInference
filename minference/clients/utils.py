import os
from openai.types.chat import ChatCompletionMessageParam

from anthropic.types import CacheControlEphemeralParam
from anthropic.types import TextBlockParam
from anthropic.types import MessageParam
from anthropic.types import ToolResultBlockParam,ToolUseBlockParam

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageParam,
)


from typing import Union, Optional, List, Tuple, Literal, Dict, Any
import json
import re
import tiktoken
import ast
import logging

logger = logging.getLogger(__name__)

def parse_json_string(self, content: str) -> Optional[Dict[str, Any]]:
    """Parse JSON string safely using multiple parsing strategies."""
    if not content or not isinstance(content, str):
        logger.debug("Empty or non-string content provided for JSON parsing")
        return None
        
    import re
    logger.debug(f"Attempting to parse JSON string (first 100 chars): {content[:100]}...")
    
    try:
        # Step 1: Extract content from common wrapper patterns
        extracted_contents = [content]  # Start with original content
        
        patterns = [
            r'<tool_call>\s*(.*?)\s*</tool_call>',
            r'<TOOL_CALL>\s*(.*?)\s*</TOOL_CALL>',
            r'\[TOOL_REQUEST\](.*?)\[END_TOOL_REQUEST\]',
            r'```(?:json)?\s*(.*?)\s*```',
            r'<tool_use>(.*?)</tool_use>'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                logger.debug(f"Extracted content using pattern: {pattern[:20]}...")
                extracted_contents.append(extracted)
        
        # Step 2: Apply various parsing strategies to each content version
        for c in extracted_contents:
            # Strategy 1: Direct JSON parsing
            try:
                result = json.loads(c)
                logger.info("Successfully parsed JSON with direct parsing")
                return result
            except json.JSONDecodeError:
                pass
            
            # Strategy 2: Python-style boolean conversion
            try:
                cleaned = re.sub(r'\bTrue\b', 'true', c)
                cleaned = re.sub(r'\bFalse\b', 'false', cleaned)
                cleaned = re.sub(r'\bNone\b', 'null', cleaned)
                result = json.loads(cleaned)
                logger.info("Successfully parsed JSON after boolean conversion")
                return result
            except json.JSONDecodeError:
                pass
            
            # Strategy 3: Extract JSON-like structure
            try:
                match = re.search(r'(\{.*\}|\[.*\])', c, re.DOTALL)
                if match:
                    result = json.loads(match.group(1))
                    logger.info("Successfully parsed JSON after extracting JSON-like structure")
                    return result
            except (json.JSONDecodeError, AttributeError):
                pass
        
        # If all strategies fail
        logger.warning("All JSON parsing strategies failed")
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error in parse_json_string: {str(e)}")
        return None
def get_ai_context_length(ai_vendor: Literal["openai", "azure_openai", "anthropic"]):
        if ai_vendor == "openai":
            return os.getenv("OPENAI_CONTEXT_LENGTH")
        if ai_vendor == "azure_openai":
            return os.getenv("AZURE_OPENAI_CONTEXT_LENGTH")
        elif ai_vendor == "anthropic":
            return os.getenv("ANTHROPIC_CONTEXT_LENGTH")


def msg_dict_to_oai(messages: List[Dict[str, Any]]) -> List[ChatCompletionMessageParam]:
        def convert_message(msg: Dict[str, Any]) -> ChatCompletionMessageParam:
            role = msg["role"]
            if role == "system":
                return ChatCompletionSystemMessageParam(role=role, content=msg["content"])
            elif role == "user":
                return ChatCompletionUserMessageParam(role=role, content=msg["content"])
            elif role == "assistant":
                assistant_msg = ChatCompletionAssistantMessageParam(role=role, content=msg.get("content"))
                if "function_call" in msg:
                    assistant_msg["function_call"] = msg["function_call"]
                if "tool_calls" in msg:
                    assistant_msg["tool_calls"] = [ChatCompletionMessageToolCallParam(**tool_call) for tool_call in msg["tool_calls"]]
                return assistant_msg
            elif role == "tool":
                return ChatCompletionToolMessageParam(role=role, content=msg["content"], tool_call_id=msg["tool_call_id"])
            elif role == "function":
                return ChatCompletionFunctionMessageParam(role=role, content=msg["content"], name=msg["name"])
            else:
                raise ValueError(f"Unknown role: {role}")

        return [convert_message(msg) for msg in messages]

def msg_dict_to_anthropic(messages: List[Dict[str, Any]],use_cache:bool=True,use_prefill:bool=False) -> Tuple[List[TextBlockParam],List[MessageParam]]:
        def create_anthropic_system_message(system_message: Optional[Dict[str, Any]],use_cache:bool=True) -> List[TextBlockParam]:
            if system_message and system_message["role"] == "system":
                text = system_message["content"]
                if use_cache:
                    return [TextBlockParam(type="text", text=text, cache_control=CacheControlEphemeralParam(type="ephemeral"))]
                else:
                    return [TextBlockParam(type="text", text=text)]
            return []

        def convert_message(msg: Dict[str, Any],use_cache:bool=False) -> Union[MessageParam, None]:
            role = msg["role"]
            content = msg["content"]
            tool_calls=msg.get("tool_calls",None)

            if role == "system":
                return None
            elif role == "tool":
                role = "user"
                if not use_cache:
                    content = [ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=msg["tool_call_id"],
                        content=content,
                        is_error=False
                    )]
                else:
                    content = [ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=msg["tool_call_id"],
                        content=content,
                        cache_control=CacheControlEphemeralParam(type="ephemeral"),
                        is_error=False
                    )]

            elif role == "assistant" and tool_calls:
               
                if not use_cache:
                
                    content = [ToolUseBlockParam(
                        type="tool_use",
                        id = tool_calls[0]["id"],
                        input = json.loads(tool_calls[0]["function"]["arguments"]),  # Parse the JSON string
                        name = tool_calls[0]["function"]["name"]
                    )]
                else:
                    content = [ToolUseBlockParam(
                        type="tool_use",
                        id = tool_calls[0]["id"],
                        input = json.loads(tool_calls[0]["function"]["arguments"]),  # Parse the JSON string
                        name = tool_calls[0]["function"]["name"],
                        cache_control=CacheControlEphemeralParam(type="ephemeral")
                    )]
                        



            elif isinstance(content, str):
                if not use_cache:
                    content = [TextBlockParam(type="text", text=content)]
                else:
                    content = [TextBlockParam(type="text", text=content,cache_control=CacheControlEphemeralParam(type='ephemeral'))]
            elif isinstance(content, list):
                if not use_cache:
                    content = [
                        TextBlockParam(type="text", text=block) if isinstance(block, str)
                        else TextBlockParam(type="text", text=block["text"]) for block in content
                    ]
                else:
                    content = [
                        TextBlockParam(type="text", text=block, cache_control=CacheControlEphemeralParam(type='ephemeral')) if isinstance(block, str)
                        else TextBlockParam(type="text", text=block["text"], cache_control=CacheControlEphemeralParam(type='ephemeral')) for block in content
                    ]
            else:
                raise ValueError("Invalid content type")
            
            return MessageParam(role=role, content=content)
        converted_messages = []
        system_message = []
        num_messages = len(messages)
        if use_cache:
            use_cache_ids = set([num_messages - 1, max(0, num_messages - 3)])
        else:
            use_cache_ids = set()
        for i,message in enumerate(messages):
            if message["role"] == "system":
                system_message= create_anthropic_system_message(message,use_cache=use_cache)
            else:
                
                use_cache_final = use_cache if  i in use_cache_ids else False
                converted_messages.append(convert_message(message,use_cache= use_cache_final))

        
        return system_message, [msg for msg in converted_messages if msg is not None]
