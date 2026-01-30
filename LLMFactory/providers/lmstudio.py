# Copyright 2023 M Chimiste

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LM Studio inference provider."""

import json
import logging
import os
import threading
import time
import warnings
from typing import List, Dict, Union, Optional, Iterator

import requests
from pydantic import BaseModel

from .base import InferenceModel, _encode_image, ThinkingResponse

logger = logging.getLogger(__name__)

# Module-level state for handling LMStudio SDK's singleton pattern.
# The SDK only allows configure_default_client() to be called once per process.
_lmstudio_configured_host: Optional[str] = None
_host_lock = threading.Lock()


# Default retry configuration for model reload
DEFAULT_MODEL_RELOAD_RETRIES = 3
DEFAULT_MODEL_RELOAD_WAIT_SECONDS = [5, 10, 15]  # Exponential backoff


def _get_configured_host() -> Optional[str]:
    """Return the currently configured LMStudio host, or None if not yet configured."""
    return _lmstudio_configured_host


def _reset_configured_host() -> None:
    """
    Reset the configured host tracking (for testing purposes only).
    
    WARNING: This does NOT reset the actual LMStudio SDK state.
    The SDK's default client cannot be reconfigured without restarting the process.
    """
    global _lmstudio_configured_host
    with _host_lock:
        _lmstudio_configured_host = None


class LMStudioInference(InferenceModel):
    """LM Studio Inference using the lmstudio-python SDK with support for remote connections and context configuration."""
    def __init__(self,
                 model_name: str,
                 max_new_tokens: int = 4096,
                 temperature: float = 0.1,
                 host: Optional[str] = None,
                 context_length: Optional[int] = None,
                 gpu_offload: Optional[Union[str, float]] = None,
                 trust_remote_code: bool = True):
        """
        Initialize LM Studio inference client.

        Args:
            model_name (str): The model identifier (e.g., "qwen2.5-7b-instruct")
            max_new_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            host (Optional[str]): Remote host (e.g., "athena.local:1234" or "192.168.1.100:1234").
                                 If None, uses localhost. Can also be set via LMSTUDIO_HOST env var.
            context_length (Optional[int]): Context window size to load model with.
                                           If None, uses model default (often 4096).
                                           Set this to use full model capacity (e.g., 32768, 131072)
            gpu_offload (Optional[Union[str, float]]): GPU offload ratio.
                                                       Can be "max" (all layers), "off" (CPU only),
                                                       or float 0-1 (proportion of layers)
            trust_remote_code (bool): Compatibility parameter (unused for LM Studio)

        Examples:
            # Local with large context
            llm = LMStudioInference("qwen2.5-7b-instruct", context_length=32768)

            # Remote with full GPU offload
            llm = LMStudioInference("llama-3.1-8b", host="athena.local:1234", gpu_offload="max")

            # Local with specific GPU ratio
            llm = LMStudioInference("mistral-7b", gpu_offload=0.5, context_length=16384)
        """
        self.host = host or os.environ.get("LMSTUDIO_HOST", "localhost:1234")
        self.context_length = context_length
        self.gpu_offload = gpu_offload
        self._model_instance = None  # Will hold the actual loaded model
        self._lms_module = None  # Will hold the lmstudio module
        super().__init__(model_name, max_new_tokens, temperature, trust_remote_code)

    def _get_provider(self) -> str:
        return "lmstudio"

    def _load_model(self):
        """Initialize the LM Studio client with singleton-aware configuration."""
        global _lmstudio_configured_host
        
        try:
            import lmstudio as lms
            self._lms_module = lms
        except ImportError:
            raise ImportError(
                "lmstudio-python is not installed. Install it with: pip install lmstudio"
            )

        # Check if remote host is reachable
        if not lms.Client.is_valid_api_host(self.host):
            raise ConnectionError(
                f"Cannot connect to LM Studio at {self.host}. "
                "Ensure LM Studio is running and network access is enabled if remote."
            )

        # Handle the SDK's singleton pattern - only configure once per process
        with _host_lock:
            if _lmstudio_configured_host is None:
                # First configuration - set the default client
                lms.configure_default_client(self.host)
                _lmstudio_configured_host = self.host
                logger.info(f"Configured LMStudio default client for host: {self.host}")
            elif _lmstudio_configured_host != self.host:
                # Different host requested - this is an SDK limitation
                warnings.warn(
                    f"LMStudio SDK limitation: Cannot change host from "
                    f"'{_lmstudio_configured_host}' to '{self.host}'. "
                    f"Using previously configured host. Restart Python process to change hosts.",
                    UserWarning
                )
                # Update self.host to match the actual configured host
                self.host = _lmstudio_configured_host
            # else: same host, already configured - nothing to do
        
        return lms

    def is_model_loaded(self) -> bool:
        """
        Check if the model is actually loaded in LM Studio.
        
        This queries the server to verify the model is in memory,
        not just that we have a cached handle.
        
        Returns:
            bool: True if the model is currently loaded, False otherwise
        """
        if self._lms_module is None:
            return False
        
        try:
            loaded_models = self._lms_module.list_loaded_models("llm")
            # Check if our model is in the loaded models list
            for model in loaded_models:
                # Model handles have an 'identifier' or we can check string representation
                model_id = getattr(model, 'identifier', None) or str(model)
                if self.model_name in model_id:
                    return True
            return False
        except Exception as e:
            logger.warning(f"Failed to check loaded models: {e}")
            return False

    def _clear_model_cache(self):
        """Clear the cached model instance, forcing a fresh load on next use."""
        self._model_instance = None
        logger.debug(f"Cleared cached model instance for {self.model_name}")

    def _get_or_load_model(self, force_reload: bool = False):
        """
        Get the model instance, loading it with config if not already loaded.
        
        Args:
            force_reload: If True, clears any cached instance and loads fresh
        """
        if force_reload:
            self._clear_model_cache()
        
        if self._model_instance is not None:
            return self._model_instance

        # Build load configuration
        # Note: LM Studio SDK uses camelCase parameter names
        config = {}
        if self.context_length is not None:
            config["contextLength"] = self.context_length
        if self.gpu_offload is not None:
            if isinstance(self.gpu_offload, str):
                # For string values like "max" or "off", use gpuOffload
                config["gpuOffload"] = self.gpu_offload
            else:
                # For numeric ratio (0-1), use gpu.ratio structure
                config["gpu"] = {"ratio": self.gpu_offload}

        # Load model with configuration (JIT loading)
        if config:
            self._model_instance = self.client.llm(self.model_name, config=config)
        else:
            self._model_instance = self.client.llm(self.model_name)
        
        logger.debug(f"Loaded model instance for {self.model_name}")
        return self._model_instance

    def wait_for_model(
        self,
        timeout_seconds: float = 30,
        poll_interval: float = 2.0
    ) -> bool:
        """
        Wait for the model to be loaded in LM Studio.
        
        Useful when LM Studio has auto-reload enabled and the model may
        take time to load after a crash or unload.
        
        Args:
            timeout_seconds: Maximum time to wait for model to load
            poll_interval: Time between status checks
            
        Returns:
            bool: True if model became available, False if timeout reached
        """
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if self.is_model_loaded():
                return True
            time.sleep(poll_interval)
        return False

    def _format_messages_for_rest(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        """Format messages into a single input string for the REST API."""
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            if role.lower() != "system":  # Skip system messages, already added
                parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    def _invoke_with_thinking(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        *,
        streaming: bool = False,
        model_name: Optional[str] = None,
        use_thinking: Union[bool, str] = True,
        return_thinking: bool = False,
        **kwargs
    ) -> Union[str, ThinkingResponse, Iterator[str]]:
        """
        Invoke LM Studio with thinking/reasoning mode using the /v1/responses REST endpoint.
        
        This method bypasses the SDK to use the REST API which supports reasoning parameters.
        """
        # Determine effort level
        if isinstance(use_thinking, str):
            effort = use_thinking
        else:
            effort = "medium"  # Default effort level
        
        # Build the REST API URL
        # Handle host format (may or may not include protocol)
        host = self.host
        if not host.startswith("http"):
            host = f"http://{host}"
        url = f"{host}/v1/responses"
        
        # Format the input
        input_text = self._format_messages_for_rest(messages, system_prompt)
        
        # Build request payload
        payload = {
            "model": model_name or self.model_name,
            "input": input_text,
            "reasoning": {"effort": effort},
            "max_tokens": kwargs.get("max_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        if streaming:
            payload["stream"] = True
            
            def _gen_thinking_stream() -> Iterator[str]:
                try:
                    with requests.post(url, json=payload, stream=True, timeout=300) as resp:
                        resp.raise_for_status()
                        thinking_parts = []
                        content_parts = []
                        
                        for line in resp.iter_lines():
                            if not line:
                                continue
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]
                                if data_str == '[DONE]':
                                    break
                                try:
                                    data = json.loads(data_str)
                                    # Handle SSE events from /v1/responses
                                    if 'output' in data:
                                        for item in data.get('output', []):
                                            item_type = item.get('type', '')
                                            for content_item in item.get('content', []):
                                                text = content_item.get('text', '')
                                                if text:
                                                    if item_type == 'reasoning':
                                                        if return_thinking:
                                                            yield f"[THINKING]{text}"
                                                    else:
                                                        yield text
                                except json.JSONDecodeError:
                                    continue
                except requests.RequestException as e:
                    raise RuntimeError(f"Error during LM Studio REST API call: {e}") from e
            
            return _gen_thinking_stream()
        else:
            # Non-streaming request
            try:
                response = requests.post(url, json=payload, timeout=300)
                response.raise_for_status()
                data = response.json()
                
                # Parse the structured response
                reasoning_text = None
                content_text = None
                
                for item in data.get('output', []):
                    item_type = item.get('type', '')
                    for content_item in item.get('content', []):
                        text = content_item.get('text', '')
                        if item_type == 'reasoning':
                            reasoning_text = text
                        elif item_type == 'message':
                            content_text = text
                
                # Fallback if content not found
                if content_text is None:
                    content_text = ""
                
                if return_thinking:
                    return ThinkingResponse(content=content_text, thinking=reasoning_text)
                else:
                    return content_text
                    
            except requests.RequestException as e:
                raise RuntimeError(f"Error during LM Studio REST API call: {e}") from e

    def invoke(self,
               messages: List[Dict[str, str]],
               system_prompt: str,
               *,
               streaming: bool = False,
               model_name: Optional[str] = None,
               schema: Optional[BaseModel] = None,
               images: Optional[List[Union[str, bytes]]] = None,
               use_thinking: Union[bool, str] = False,
               return_thinking: bool = False,
               **kwargs) -> Union[str, ThinkingResponse, Iterator[str]]:
        """
        Invoke the LM Studio model to generate a response.

        Args:
            messages (List[Dict[str, str]]): Conversation history as list of dicts with 'role' and 'content'
            system_prompt (str): System prompt to guide model behavior
            streaming (bool): Whether to stream the response token by token
            model_name (Optional[str]): Override model name (loads different model if specified)
            schema (Optional[BaseModel]): Pydantic schema for structured JSON output
            images (Optional[List[Union[str, bytes]]]): Images for multimodal models (VLMs).
                                                        Can be file paths or raw bytes.
            use_thinking (Union[bool, str], optional): Enable thinking/reasoning mode.
                Can be True (uses "medium" effort), False (disabled), or a string
                "low"/"medium"/"high" to specify effort level. When enabled, uses
                the /v1/responses REST endpoint. Defaults to False.
            return_thinking (bool, optional): If True and use_thinking is enabled,
                return a ThinkingResponse containing both thinking trace and content.
                If False, return only the content. Defaults to False.
            **kwargs: Additional parameters:
                     - max_tokens: Override max_new_tokens
                     - temperature: Override temperature
                     - top_p, top_k, stop: Sampling parameters

        Returns:
            Union[str, ThinkingResponse, Iterator[str]]: Full response string, ThinkingResponse
                (if return_thinking=True), or iterator yielding tokens if streaming=True

        Examples:
            # Simple query
            response = llm.invoke(
                [{"role": "user", "content": "Hello!"}],
                "You are a helpful assistant."
            )

            # Streaming
            for token in llm.invoke(messages, system, streaming=True):
                print(token, end="", flush=True)

            # With thinking/reasoning mode
            response = llm.invoke(
                [{"role": "user", "content": "Solve: 15 * 23"}],
                "You are a math tutor.",
                use_thinking="high",
                return_thinking=True
            )
            print(f"Thinking: {response.thinking}")
            print(f"Answer: {response.content}")

            # With images (VLM)
            response = llm.invoke(
                [{"role": "user", "content": "What's in this image?"}],
                "You are a vision assistant.",
                images=["photo.jpg"]
            )

            # Structured output
            from pydantic import BaseModel
            class Answer(BaseModel):
                answer: str
                confidence: float

            response = llm.invoke(messages, system, schema=Answer)
        """
        # If thinking mode is enabled, use the REST API endpoint
        if use_thinking:
            if images:
                logger.warning("Images are not supported with thinking mode. Ignoring images.")
            if schema:
                logger.warning("Schema is not supported with thinking mode. Ignoring schema.")
            return self._invoke_with_thinking(
                messages=messages,
                system_prompt=system_prompt,
                streaming=streaming,
                model_name=model_name,
                use_thinking=use_thinking,
                return_thinking=return_thinking,
                **kwargs
            )
        
        # Standard path using the SDK
        # Get or load the model
        model = self._get_or_load_model()

        # Create Chat object with system prompt as constructor argument
        chat = self.client.Chat(system_prompt)
        
        # Process messages and handle images
        image_handles = None
        if images:
            # Prepare image handles using LM Studio's prepare_image
            image_handles = []
            for img in images:
                if isinstance(img, bytes):
                    # Pass bytes directly
                    image_handles.append(self.client.prepare_image(img))
                elif isinstance(img, str):
                    # Path to image file
                    image_handles.append(self.client.prepare_image(img))
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}. Use str (path) or bytes.")

        # Add conversation history
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # If this is the last user message and we have images, add them
            is_last_user_msg = (i == len(messages) - 1 and role == "user")

            if role == "system":
                # Skip system messages as they're handled in Chat constructor
                continue
            elif role == "assistant":
                chat.add_assistant_message(content)
            else:  # user or any other role
                if is_last_user_msg and image_handles:
                    chat.add_user_message(content, images=image_handles)
                else:
                    chat.add_user_message(content)

        # Build generation config
        # Note: LM Studio SDK uses a 'config' dict with camelCase parameter names
        config = {
            "temperature": kwargs.get("temperature", self.temperature),
            "maxTokens": kwargs.get("max_tokens", self.max_new_tokens),
        }
        
        # Add structured output support if schema provided
        if schema:
            config["response_format"] = schema

        # Add additional sampling parameters if provided
        # Map snake_case to camelCase for LM Studio SDK
        if "top_p" in kwargs:
            config["topP"] = kwargs["top_p"]
        if "top_k" in kwargs:
            config["topK"] = kwargs["top_k"]
        if "stop" in kwargs:
            config["stop"] = kwargs["stop"]

        # Retry logic for model-not-found errors (model may have crashed/unloaded)
        max_retries = kwargs.get('_model_reload_retries', DEFAULT_MODEL_RELOAD_RETRIES)
        wait_times = kwargs.get('_model_reload_wait_seconds', DEFAULT_MODEL_RELOAD_WAIT_SECONDS)
        
        last_error = None
        for attempt in range(max_retries):
            try:
                if streaming:
                    # Return streaming iterator
                    stream = model.respond_stream(chat, config=config)

                    def _gen() -> Iterator[str]:
                        for chunk in stream:
                            # Handle different possible response formats
                            if hasattr(chunk, 'content') and chunk.content:
                                yield chunk.content
                            elif isinstance(chunk, dict) and 'content' in chunk:
                                if chunk['content']:
                                    yield chunk['content']
                            elif isinstance(chunk, str) and chunk:
                                yield chunk
                        
                        # For structured output with streaming, get final parsed result
                        if schema:
                            try:
                                final_result = stream.result()
                                if hasattr(final_result, 'parsed'):
                                    # Store parsed result for access after iteration
                                    _gen.parsed_result = final_result.parsed
                            except Exception:
                                pass

                    return _gen()
                else:
                    # Return full response
                    response = model.respond(chat, config=config)

                    # For structured output, return parsed result
                    if schema and hasattr(response, 'parsed'):
                        return response.parsed
                    
                    # Handle different possible response formats
                    if hasattr(response, 'content'):
                        return response.content
                    elif isinstance(response, dict) and 'content' in response:
                        return response['content']
                    else:
                        return str(response)

            except Exception as e:
                error_str = str(e).lower()
                last_error = e
                
                # Check if this is a model-not-found or model-crashed error
                is_model_error = (
                    'LMStudioModelNotFoundError' in type(e).__name__ or
                    'no model found' in error_str or
                    'nomodelmatchingquery' in error_str or
                    'model has crashed' in error_str or
                    'totalloadedmodels' in error_str and '0' in error_str
                )
                
                if is_model_error and attempt < max_retries - 1:
                    # Clear stale model handle
                    self._clear_model_cache()
                    
                    # Wait for potential auto-reload
                    wait_time = wait_times[min(attempt, len(wait_times) - 1)]
                    logger.warning(
                        f"Model '{self.model_name}' appears unloaded (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {wait_time}s for potential reload..."
                    )
                    time.sleep(wait_time)
                    
                    # Verify model is loaded before retrying
                    if not self.is_model_loaded():
                        logger.warning(f"Model '{self.model_name}' still not loaded after waiting")
                        # Try to trigger a fresh load
                        try:
                            model = self._get_or_load_model(force_reload=True)
                        except Exception as load_err:
                            logger.warning(f"Failed to reload model: {load_err}")
                            continue
                    else:
                        logger.info(f"Model '{self.model_name}' is now loaded, retrying...")
                        model = self._get_or_load_model(force_reload=True)
                    
                    continue
                else:
                    # Non-recoverable error or max retries exceeded
                    break
        
        # If we get here, all retries failed
        raise RuntimeError(
            f"Error during LM Studio inference: {str(last_error)}. "
            f"Check that model '{self.model_name}' is available and context length is appropriate."
        ) from last_error

    def verify(self, wait_for_load: bool = False, timeout: float = 30) -> dict:
        """
        Verify connection to LM Studio and check model load status.
        
        Unlike _load_model() which only checks connectivity, this method
        also verifies whether the specified model is actually loaded in memory.
        
        Args:
            wait_for_load: If True, wait for model to become available
            timeout: Timeout in seconds if wait_for_load is True
            
        Returns:
            dict: Status information including:
                - connected (bool): Whether LM Studio server is reachable
                - model_loaded (bool): Whether the specified model is loaded
                - loaded_models (list): List of currently loaded model identifiers
                - error (str|None): Error message if verification failed
        """
        result = {
            'connected': False,
            'model_loaded': False,
            'loaded_models': [],
            'error': None
        }
        
        try:
            # Check connectivity
            if self._lms_module is None:
                self._load_model()
            
            if not self._lms_module.Client.is_valid_api_host(self.host):
                result['error'] = f"Cannot connect to LM Studio at {self.host}"
                return result
            
            result['connected'] = True
            
            # Get loaded models
            try:
                loaded_models = self._lms_module.list_loaded_models("llm")
                result['loaded_models'] = [
                    getattr(m, 'identifier', str(m)) for m in loaded_models
                ]
            except Exception as e:
                result['error'] = f"Failed to list loaded models: {e}"
                return result
            
            # Check if our model is loaded
            model_loaded = self.is_model_loaded()
            
            if not model_loaded and wait_for_load:
                logger.info(f"Waiting up to {timeout}s for model '{self.model_name}' to load...")
                model_loaded = self.wait_for_model(timeout_seconds=timeout)
                if model_loaded:
                    # Refresh loaded models list
                    loaded_models = self._lms_module.list_loaded_models("llm")
                    result['loaded_models'] = [
                        getattr(m, 'identifier', str(m)) for m in loaded_models
                    ]
            
            result['model_loaded'] = model_loaded
            
            if not model_loaded:
                result['error'] = (
                    f"Model '{self.model_name}' is not loaded. "
                    f"Loaded models: {result['loaded_models'] or 'none'}"
                )
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            return result

    def ensure_model_loaded(self, timeout: float = 60) -> bool:
        """
        Ensure the model is loaded and ready for inference.
        
        This is a convenience method that:
        1. Checks if model is currently loaded
        2. If not, waits for auto-reload (if LM Studio has it enabled)
        3. Clears any stale cached handles
        4. Returns whether the model is ready
        
        Args:
            timeout: Maximum time to wait for model to become available
            
        Returns:
            bool: True if model is loaded and ready, False otherwise
        """
        # First check current status
        if self.is_model_loaded():
            return True
        
        # Clear any stale handle
        self._clear_model_cache()
        
        # Wait for model to become available
        logger.info(f"Model '{self.model_name}' not loaded, waiting up to {timeout}s...")
        if self.wait_for_model(timeout_seconds=timeout):
            logger.info(f"Model '{self.model_name}' is now loaded")
            return True
        
        logger.warning(f"Model '{self.model_name}' did not load within {timeout}s")
        return False

    def unload_model(self):
        """Unload the model from memory to free resources."""
        if self._model_instance is not None:
            try:
                self._model_instance.unload()
            except Exception:
                pass  # Ignore errors during unload
            self._model_instance = None

    def close(self):
        """Clean up resources."""
        self.unload_model()

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.close()
