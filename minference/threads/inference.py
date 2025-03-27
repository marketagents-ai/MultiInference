"""
Refactored InferenceOrchestrator using utility functions for payload handling.
All imports properly managed and type safety enforced.
"""
import asyncio
import json
import os
from typing import List, Dict, Any, Optional, Literal
from pydantic import Field
from dotenv import load_dotenv
import time
from uuid import UUID, uuid4
import aiofiles

# Internal imports - complete set
from minference.threads.models import (
    RawOutput, ProcessedOutput, ChatThread, LLMClient,
    ResponseFormat, Entity, ChatMessage,
    MessageRole
)
from minference.oai_parallel import (
    process_api_requests_from_file,
    OAIApiFromFileConfig
)
from minference.clients.requests import (
    prepare_requests_file,
    convert_chat_thread_to_request,
    create_oai_completion_config,
    create_anthropic_completion_config,
    create_vllm_completion_config,
    create_litellm_completion_config,
    create_openrouter_completion_config
)
from minference.ecs.entity import EntityRegistry, entity_tracer


class RequestLimits(Entity):
    """
    Configuration for API request limits.
    Inherits from Entity for UUID handling and registry integration.
    """
    max_requests_per_minute: int = Field(
        default=50,
        description="The maximum number of requests per minute for the API"
    )
    max_tokens_per_minute: int = Field(
        default=100000,
        description="The maximum number of tokens per minute for the API"
    )
    provider: Literal["openai", "anthropic", "vllm", "litellm", "openrouter"] = Field(
        default="openai",
        description="The provider of the API"
    )

def create_chat_thread_hashmap(chat_threads: List[ChatThread]) -> Dict[UUID, ChatThread]:
    """Create a hashmap of chat threads by their IDs."""
    return {p.id: p for p in chat_threads if p.id is not None}

async def process_outputs_and_execute_tools(chat_threads: List[ChatThread], llm_outputs: List[ProcessedOutput]) -> List[ProcessedOutput]:
    """Process outputs and execute tools in parallel."""
    # Track thread ID mappings (original -> latest)
    thread_id_mappings = {chat_thread.live_id: chat_thread for chat_thread in chat_threads}
    history_update_tasks = []
    
    EntityRegistry._logger.info("""
=================================================================
                    START PROCESSING OUTPUTS 
=================================================================
""")
    
    for output in llm_outputs:
        if output.chat_thread_live_id:
            
            try:
                # Get the latest version from the thread mapping
                chat_thread = thread_id_mappings.get(output.chat_thread_live_id,None)
                if not chat_thread:
                    EntityRegistry._logger.error(f"ChatThread({output.chat_thread_id}) not found in temporary thread_id_mappings")
                    continue
                    
                EntityRegistry._logger.info(f"""
=== PROCESSING OUTPUT FOR CHAT THREAD ===
Current Thread ID: {chat_thread.id}
Current History Length: {len(chat_thread.history)}
Current New Message: {chat_thread.new_message}
Current Parent ID: {chat_thread.parent_id}
Current Lineage ID: {chat_thread.lineage_id}
=======================================
""")
                
                # Add history and get potentially new version
                history_update_tasks.append(
                    chat_thread.add_chat_turn_history(output)
                )
                EntityRegistry._logger.info(
                    f"Queued history update for ChatThread({chat_thread.id})"
                )
            except Exception as e:
                EntityRegistry._logger.error(f"""
=== ERROR PROCESSING CHAT THREAD ===
Thread ID: {output.chat_thread_id}
Error: {str(e)}
Traceback: {e.__traceback__}
===================================
""")
    
    if history_update_tasks:
        EntityRegistry._logger.info(f"""
=================================================================
            EXECUTING {len(history_update_tasks)} HISTORY UPDATES 
=================================================================
""")
        try:
            results = await asyncio.gather(*history_update_tasks)
            
            
        except Exception as e:
            EntityRegistry._logger.error(f"""
=== ERROR DURING HISTORY UPDATES ===
Error: {str(e)}
Traceback: {e.__traceback__}
==================================
""")
    
    return llm_outputs

async def run_parallel_ai_completion(
    chat_threads: List[ChatThread],
    orchestrator: 'InferenceOrchestrator'
) -> List[ProcessedOutput]:
    """Run parallel AI completion for multiple chat threads."""
    EntityRegistry._logger.info(f"Starting parallel AI completion for {len(chat_threads)} chat threads")
    
    # Track original to forked thread mappings
    original_ids = [chat.id for chat in chat_threads]
    thread_mappings = {}
    
    # First add user messages to all chat threads
    for chat in chat_threads:
        try:
            EntityRegistry._logger.info(f"Adding user message to ChatThread({chat.id})")
            original_id = chat.id
            result = chat.add_user_message()
           
        except Exception as e:
            if chat.llm_config.response_format != ResponseFormat.auto_tools or chat.llm_config.response_format != ResponseFormat.workflow:
                chat_threads.remove(chat)
                EntityRegistry._logger.error(f"Error adding user message to ChatThread({chat.id}): {e}")

    # Run LLM completions in parallel
    tasks = []
    if any(p for p in chat_threads if p.llm_config.client == "openai"):
        tasks.append(orchestrator._run_openai_completion([p for p in chat_threads if p.llm_config.client == "openai"]))
    if any(p for p in chat_threads if p.llm_config.client == "anthropic"):
        tasks.append(orchestrator._run_anthropic_completion([p for p in chat_threads if p.llm_config.client == "anthropic"]))
    if any(p for p in chat_threads if p.llm_config.client == "vllm"):
        tasks.append(orchestrator._run_vllm_completion([p for p in chat_threads if p.llm_config.client == "vllm"]))
    if any(p for p in chat_threads if p.llm_config.client == "litellm"):
        tasks.append(orchestrator._run_litellm_completion([p for p in chat_threads if p.llm_config.client == "litellm"]))
    if any(p for p in chat_threads if p.llm_config.client == "openrouter"):
        tasks.append(orchestrator._run_openrouter_completion([p for p in chat_threads if p.llm_config.client == "openrouter"]))

    results = await asyncio.gather(*tasks)
    llm_outputs = [item for sublist in results for item in sublist]
    
    

    
    EntityRegistry._logger.info(f"Processing {len(llm_outputs)} LLM outputs")
    processed_outputs = await process_outputs_and_execute_tools(chat_threads, llm_outputs)


    
    return processed_outputs

async def parse_results_file(filepath: str, client: LLMClient) -> List[ProcessedOutput]:
    """Parse results file and convert to ProcessedOutput objects asynchronously."""
    results = []
    pending_tasks = []
    EntityRegistry._logger.info(f"Parsing results from {filepath}")
    
    async def process_line(line: str) -> Optional[ProcessedOutput]:
        """Process a single line asynchronously."""
        try:
            result = json.loads(line)
            processed_output = await convert_result_to_llm_output(result, client)
            return processed_output
        except json.JSONDecodeError:
            EntityRegistry._logger.error(f"Error decoding JSON: {line}")
        except Exception as e:
            EntityRegistry._logger.error(f"Error processing result: {e}")
        return None

    async with aiofiles.open(filepath, 'r') as f:
        async for line in f:
            # Create task for each line
            task = asyncio.create_task(process_line(line))
            pending_tasks.append(task)
    
    # Wait for all tasks to complete
    completed_tasks = await asyncio.gather(*pending_tasks)
    
    # Filter out None results and add to results list
    results = [result for result in completed_tasks if result is not None]
    
    EntityRegistry._logger.info(f"Processed {len(results)} results from {filepath}")
    return results

async def convert_result_to_llm_output(result: List[Dict[str, Any]], client: LLMClient) -> ProcessedOutput:
    """Convert raw result directly to ProcessedOutput."""
    metadata, request_data, response_data = result
    # EntityRegistry._logger.info(f"Converting result for chat_thread_id: {metadata['chat_thread_id']}")

    raw_output = RawOutput(
        raw_result=response_data,
        completion_kwargs=request_data,
        start_time=metadata["start_time"],
        end_time=metadata["end_time"] or time.time(),
        chat_thread_id=metadata["chat_thread_id"],
        chat_thread_live_id=metadata["chat_thread_live_id"],
        client=client
    )

    return raw_output.create_processed_output()


class InferenceOrchestrator:
    def __init__(self, 
                 oai_request_limits: Optional[RequestLimits] = None, 
                 anthropic_request_limits: Optional[RequestLimits] = None, 
                 vllm_request_limits: Optional[RequestLimits] = None,
                 litellm_request_limits: Optional[RequestLimits] = None,
                 openrouter_request_limits: Optional[RequestLimits] = None,
                 local_cache: bool = True,
                 cache_folder: Optional[str] = None,
                 vllm_endpoint:Optional[str] = None):
        load_dotenv()
        EntityRegistry._logger.info("Initializing InferenceOrchestrator")
        
        # API Keys and Endpoints
        self.openai_key = os.getenv("OPENAI_KEY", "")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.vllm_key = os.getenv("VLLM_API_KEY", "")
        if vllm_endpoint:
            self.default_vllm_endpoint = vllm_endpoint
        else:
            self.default_vllm_endpoint = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
        self.default_litellm_endpoint = os.getenv("LITELLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
        self.litellm_key = os.getenv("LITELLM_API_KEY", "")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        self.default_openrouter_endpoint = os.getenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions")
        
        # Request Limits
        self.oai_request_limits = oai_request_limits or RequestLimits(
            max_requests_per_minute=500,
            max_tokens_per_minute=200000,
            provider="openai"
        )
        self.anthropic_request_limits = anthropic_request_limits or RequestLimits(
            max_requests_per_minute=50,
            max_tokens_per_minute=40000,
            provider="anthropic"
        )
        self.vllm_request_limits = vllm_request_limits or RequestLimits(
            max_requests_per_minute=500,
            max_tokens_per_minute=200000,
            provider="vllm"
        )
        self.litellm_request_limits = litellm_request_limits or RequestLimits(
            max_requests_per_minute=500,
            max_tokens_per_minute=200000,
            provider="litellm"
        )
        self.openrouter_request_limits = openrouter_request_limits or RequestLimits(
            max_requests_per_minute=50,
            max_tokens_per_minute=40000,
            provider="openrouter"
        )
        
        # Cache setup
        self.local_cache = local_cache
        self.cache_folder = self._setup_cache_folder(cache_folder)
        self.all_requests = []
        EntityRegistry._logger.info("InferenceOrchestrator initialized")

    def _setup_cache_folder(self, cache_folder: Optional[str]) -> str:
        if cache_folder:
            full_path = os.path.abspath(cache_folder)
        else:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            full_path = os.path.join(repo_root, 'outputs', 'inference_cache')
        
        os.makedirs(full_path, exist_ok=True)
        EntityRegistry._logger.info(f"Cache folder set up at: {full_path}")
        return full_path
    
    def _create_chat_thread_hashmap(self, chat_threads: List[ChatThread]) -> Dict[UUID, ChatThread]:
        """Create a hashmap of chat threads by their IDs."""
        return create_chat_thread_hashmap(chat_threads)

    async def run_parallel_ai_completion(self, chat_threads: List[ChatThread]) -> List[ProcessedOutput]:
        """Run parallel AI completion for multiple chat threads."""
        return await run_parallel_ai_completion(chat_threads, self)

    async def _process_outputs_and_execute_tools(self, chat_threads: List[ChatThread], llm_outputs: List[ProcessedOutput]) -> List[ProcessedOutput]:
        """Process outputs and execute tools in parallel."""
        return await process_outputs_and_execute_tools(chat_threads, llm_outputs)

    async def _run_openai_completion(self, chat_threads: List[ChatThread]) -> List[ProcessedOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        unique_uuid = str(uuid4())
        requests_file = os.path.join(self.cache_folder, f'openai_requests_{unique_uuid}_{timestamp}.jsonl')

        results_file = os.path.join(self.cache_folder, f'openai_results_{unique_uuid}_{timestamp}.jsonl')
        
        prepare_requests_file(chat_threads, "openai", requests_file)

        config = create_oai_completion_config(
            chat_thread=chat_threads[0], 
            requests_file=requests_file, 
            results_file=results_file,
            openai_key=self.openai_key,
            max_requests_per_minute=self.oai_request_limits.max_requests_per_minute,
            max_tokens_per_minute=self.oai_request_limits.max_tokens_per_minute
        )
        
        if config:
            try:
                await process_api_requests_from_file(config)
                return await parse_results_file(results_file, client=LLMClient.openai)
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []

    async def _run_anthropic_completion(self, chat_threads: List[ChatThread]) -> List[ProcessedOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        unique_uuid = str(uuid4())
        requests_file = os.path.join(self.cache_folder, f'anthropic_requests_{unique_uuid}_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'anthropic_results_{unique_uuid}_{timestamp}.jsonl')
        

        prepare_requests_file(chat_threads, "anthropic", requests_file)
        config = create_anthropic_completion_config(
            chat_thread=chat_threads[0], 
            requests_file=requests_file, 
            results_file=results_file,
            anthropic_key=self.anthropic_key,
            max_requests_per_minute=self.anthropic_request_limits.max_requests_per_minute,
            max_tokens_per_minute=self.anthropic_request_limits.max_tokens_per_minute
        )
        
        if config:
            try:
                await process_api_requests_from_file(config)
                return await parse_results_file(results_file, client=LLMClient.anthropic)
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []

    async def _run_vllm_completion(self, chat_threads: List[ChatThread]) -> List[ProcessedOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        unique_uuid = str(uuid4())
        requests_file = os.path.join(self.cache_folder, f'vllm_requests_{unique_uuid}_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'vllm_results_{unique_uuid}_{timestamp}.jsonl')
        

        prepare_requests_file(chat_threads, "vllm", requests_file)
        config = create_vllm_completion_config(
            chat_thread=chat_threads[0], 
            requests_file=requests_file, 
            results_file=results_file,
            vllm_endpoint=self.default_vllm_endpoint,
            vllm_key=self.vllm_key,
            max_requests_per_minute=self.vllm_request_limits.max_requests_per_minute,
            max_tokens_per_minute=self.vllm_request_limits.max_tokens_per_minute
        )
        
        if config:
            try:
                await process_api_requests_from_file(config)
                return await parse_results_file(results_file, client=LLMClient.vllm)
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []

    async def _run_litellm_completion(self, chat_threads: List[ChatThread]) -> List[ProcessedOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        unique_uuid = str(uuid4())
        requests_file = os.path.join(self.cache_folder, f'litellm_requests_{unique_uuid}_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'litellm_results_{unique_uuid}_{timestamp}.jsonl')
        

        prepare_requests_file(chat_threads, "litellm", requests_file)
        config = create_litellm_completion_config(
            chat_thread=chat_threads[0], 
            requests_file=requests_file, 
            results_file=results_file,
            litellm_endpoint=self.default_litellm_endpoint,
            litellm_key=self.litellm_key,
            max_requests_per_minute=self.litellm_request_limits.max_requests_per_minute,
            max_tokens_per_minute=self.litellm_request_limits.max_tokens_per_minute
        )
        
        if config:
            try:
                await process_api_requests_from_file(config)
                return await parse_results_file(results_file, client=LLMClient.litellm)
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []

    async def _run_openrouter_completion(self, chat_threads: List[ChatThread]) -> List[ProcessedOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        unique_uuid = str(uuid4())
        requests_file = os.path.join(self.cache_folder, f'openrouter_requests_{unique_uuid}_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'openrouter_results_{unique_uuid}_{timestamp}.jsonl')
        
        prepare_requests_file(chat_threads, "openrouter", requests_file)
        config = create_openrouter_completion_config(
            chat_thread=chat_threads[0], 
            requests_file=requests_file, 
            results_file=results_file,
            openrouter_endpoint=self.default_openrouter_endpoint,
            openrouter_key=self.openrouter_key,
            max_requests_per_minute=self.openrouter_request_limits.max_requests_per_minute,
            max_tokens_per_minute=self.openrouter_request_limits.max_tokens_per_minute
        )
        
        if config:
            try:
                await process_api_requests_from_file(config)
                return await parse_results_file(results_file, client=LLMClient.openrouter)
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []

    def _delete_files(self, *files):
        """Delete temporary files if not caching locally."""
        for file in files:
            try:
                os.remove(file)
                EntityRegistry._logger.info(f"Deleted file: {file}")
            except OSError as e:
                EntityRegistry._logger.error(f"Error deleting file {file}: {e}")