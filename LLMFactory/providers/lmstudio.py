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

"""LM Studio inference provider using OpenAI-compatible API.

This provider uses the OpenAI Python SDK to communicate with LM Studio's
OpenAI-compatible endpoints, providing a more stable and well-tested
integration than the native lmstudio-python SDK.
"""

import json
import logging
import os
import time
from typing import List, Dict, Union, Optional, Iterator

import requests
from openai import OpenAI
from pydantic import BaseModel

from .base import InferenceModel, _encode_image, ThinkingResponse

logger = logging.getLogger(__name__)

# Default retry configuration for model reload
DEFAULT_MODEL_RELOAD_RETRIES = 3
DEFAULT_MODEL_RELOAD_WAIT_SECONDS = [5, 10, 15]  # Exponential backoff


class LMStudioInference(InferenceModel):
    """LM Studio Inference using OpenAI-compatible API.
    
    This provider uses the OpenAI Python SDK to communicate with LM Studio,
    providing better stability than the native lmstudio-python SDK.
    Model management (load/unload/status) uses direct HTTP calls to 
    LM Studio's REST API.
    """
    
    def __init__(self,
                 model_name: str,
                 max_new_tokens: int = 4096,
                 temperature: float = 0.1,
                 host: Optional[str] = None,
                 context_length: Optional[int] = None,
                 gpu_offload: Optional[Union[str, float]] = None,
                 flash_attention: Optional[bool] = None,
                 trust_remote_code: bool = True,
                 api_key: str = "lm-studio"):
        """
        Initialize LM Studio inference client.

        Args:
            model_name (str): The model identifier (e.g., "qwen2.5-7b-instruct")
            max_new_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            host (Optional[str]): Host address (e.g., "localhost:1234" or "athena.local:1234").
                                 If None, uses localhost:1234. Can also be set via LMSTUDIO_HOST env var.
            context_length (Optional[int]): Context window size to load model with.
                                           Used when auto-loading models via API.
            gpu_offload (Optional[Union[str, float]]): GPU offload setting for model loading.
            flash_attention (Optional[bool]): Enable flash attention when loading model.
            trust_remote_code (bool): Compatibility parameter (unused for LM Studio)
            api_key (str): API key for LM Studio (default "lm-studio", usually not needed)

        Examples:
            # Basic usage
            llm = LMStudioInference("qwen2.5-7b-instruct")

            # Remote server
            llm = LMStudioInference("llama-3.1-8b", host="athena.local:1234")

            # With specific context length for loading
            llm = LMStudioInference("mistral-7b", context_length=32768)
        """
        self.host = host or os.environ.get("LMSTUDIO_HOST", "localhost:1234")
        self.context_length = context_length
        self.gpu_offload = gpu_offload
        self.flash_attention = flash_attention
        self.api_key = api_key
        
        # Ensure host has protocol
        if not self.host.startswith("http"):
            self.base_url = f"http://{self.host}"
        else:
            self.base_url = self.host
            # Strip protocol for self.host to keep just host:port
            self.host = self.host.replace("http://", "").replace("https://", "")
        
        super().__init__(model_name, max_new_tokens, temperature, trust_remote_code)

    def _get_provider(self) -> str:
        return "lmstudio"

    def _load_model(self):
        """Initialize the OpenAI client pointing to LM Studio."""
        # Verify connectivity
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to LM Studio at {self.base_url}. "
                f"Ensure LM Studio is running and the server is enabled. Error: {e}"
            )
        
        return OpenAI(base_url=f"{self.base_url}/v1", api_key=self.api_key)

    def _get_native_api_url(self, endpoint: str) -> str:
        """Get the full URL for native LM Studio API endpoints."""
        return f"{self.base_url}/api/v1{endpoint}"

    def is_model_loaded(self) -> bool:
        """
        Check if the model is currently loaded in LM Studio.
        
        Returns:
            bool: True if the model is loaded, False otherwise
        """
        try:
            response = requests.get(
                self._get_native_api_url("/models"),
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            for model in data.get("models", []):
                model_key = model.get("key", "")
                loaded_instances = model.get("loaded_instances", [])
                
                # Check if our model matches and has loaded instances
                if self.model_name in model_key and len(loaded_instances) > 0:
                    return True
                    
                # Also check instance IDs directly
                for instance in loaded_instances:
                    if self.model_name in instance.get("id", ""):
                        return True
            
            return False
        except Exception as e:
            logger.warning(f"Failed to check model load status: {e}")
            return False

    def get_loaded_models(self) -> List[Dict]:
        """
        Get list of currently loaded models with their configurations.
        
        Returns:
            List[Dict]: List of loaded model info dictionaries
        """
        try:
            response = requests.get(
                self._get_native_api_url("/models"),
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            loaded = []
            for model in data.get("models", []):
                instances = model.get("loaded_instances", [])
                if instances:
                    loaded.append({
                        "key": model.get("key"),
                        "display_name": model.get("display_name"),
                        "type": model.get("type"),
                        "instances": instances
                    })
            return loaded
        except Exception as e:
            logger.warning(f"Failed to get loaded models: {e}")
            return []

    def load_model_explicit(
        self,
        context_length: Optional[int] = None,
        flash_attention: Optional[bool] = None,
        timeout: float = 120
    ) -> bool:
        """
        Explicitly load the model into LM Studio memory.
        
        Args:
            context_length: Context length to use (overrides instance setting)
            flash_attention: Enable flash attention (overrides instance setting)
            timeout: Maximum time to wait for model to load
            
        Returns:
            bool: True if model loaded successfully
        """
        payload = {"model": self.model_name}
        
        ctx_len = context_length or self.context_length
        if ctx_len:
            payload["context_length"] = ctx_len
        
        flash = flash_attention if flash_attention is not None else self.flash_attention
        if flash is not None:
            payload["flash_attention"] = flash
            
        try:
            response = requests.post(
                self._get_native_api_url("/models/load"),
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            logger.info(
                f"Model {self.model_name} loaded in {result.get('load_time_seconds', '?')}s"
            )
            return True
        except requests.Timeout:
            logger.error(f"Timeout loading model {self.model_name}")
            return False
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            return False

    def unload_model(self) -> bool:
        """
        Unload the model from LM Studio memory.
        
        Returns:
            bool: True if model unloaded successfully
        """
        try:
            response = requests.post(
                self._get_native_api_url("/models/unload"),
                json={"model": self.model_name},
                timeout=30
            )
            response.raise_for_status()
            logger.info(f"Model {self.model_name} unloaded")
            return True
        except Exception as e:
            logger.warning(f"Failed to unload model {self.model_name}: {e}")
            return False

    def wait_for_model(
        self,
        timeout_seconds: float = 60,
        poll_interval: float = 2.0
    ) -> bool:
        """
        Wait for the model to be loaded in LM Studio.
        
        Args:
            timeout_seconds: Maximum time to wait
            poll_interval: Time between status checks
            
        Returns:
            bool: True if model became available, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if self.is_model_loaded():
                return True
            time.sleep(poll_interval)
        return False

    def ensure_model_loaded(self, timeout: float = 120) -> bool:
        """
        Ensure the model is loaded, loading it if necessary.
        
        Args:
            timeout: Maximum time to wait for loading
            
        Returns:
            bool: True if model is ready for inference
        """
        if self.is_model_loaded():
            return True
        
        logger.info(f"Model {self.model_name} not loaded, attempting to load...")
        
        # Try to load the model
        if self.load_model_explicit(timeout=timeout):
            return True
        
        # If explicit load failed, wait in case LM Studio is auto-loading
        logger.info(f"Waiting for model {self.model_name} to become available...")
        return self.wait_for_model(timeout_seconds=timeout)

    def verify(self, wait_for_load: bool = False, timeout: float = 30) -> dict:
        """
        Verify connection to LM Studio and check model status.
        
        Args:
            wait_for_load: If True, wait for model to become available
            timeout: Timeout in seconds if wait_for_load is True
            
        Returns:
            dict: Status information including:
                - connected (bool): Whether LM Studio is reachable
                - model_loaded (bool): Whether the model is loaded
                - loaded_models (list): List of currently loaded models
                - error (str|None): Error message if any
        """
        result = {
            'connected': False,
            'model_loaded': False,
            'loaded_models': [],
            'error': None
        }
        
        try:
            # Check connectivity
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            response.raise_for_status()
            result['connected'] = True
            
            # Get loaded models
            loaded = self.get_loaded_models()
            result['loaded_models'] = [m.get('key') for m in loaded]
            
            # Check if our model is loaded
            model_loaded = self.is_model_loaded()
            
            if not model_loaded and wait_for_load:
                logger.info(f"Waiting for model '{self.model_name}' to load...")
                model_loaded = self.wait_for_model(timeout_seconds=timeout)
                if model_loaded:
                    loaded = self.get_loaded_models()
                    result['loaded_models'] = [m.get('key') for m in loaded]
            
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

    def _format_messages_for_responses(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: str
    ) -> str:
        """Format messages for the /v1/responses endpoint."""
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            if role.lower() != "system":
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
        Invoke with thinking/reasoning mode using /v1/responses endpoint.
        """
        # Determine effort level
        if isinstance(use_thinking, str):
            effort = use_thinking
        else:
            effort = "medium"
        
        url = f"{self.base_url}/v1/responses"
        input_text = self._format_messages_for_responses(messages, system_prompt)
        
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
                    raise RuntimeError(f"Error during LM Studio thinking API call: {e}") from e
            
            return _gen_thinking_stream()
        else:
            try:
                response = requests.post(url, json=payload, timeout=300)
                response.raise_for_status()
                data = response.json()
                
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
                
                if content_text is None:
                    content_text = ""
                
                if return_thinking:
                    return ThinkingResponse(content=content_text, thinking=reasoning_text)
                else:
                    return content_text
                    
            except requests.RequestException as e:
                raise RuntimeError(f"Error during LM Studio thinking API call: {e}") from e

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
            messages (List[Dict[str, str]]): Conversation history as list of 
                dicts with 'role' and 'content'
            system_prompt (str): System prompt to guide model behavior
            streaming (bool): Whether to stream the response token by token
            model_name (Optional[str]): Override model name for this call
            schema (Optional[BaseModel]): Pydantic schema for structured JSON output
            images (Optional[List[Union[str, bytes]]]): Images for vision models.
                Can be file paths or raw bytes.
            use_thinking (Union[bool, str]): Enable thinking/reasoning mode.
                Can be True (uses "medium" effort), False (disabled), or a string
                "low"/"medium"/"high" to specify effort level.
            return_thinking (bool): If True and use_thinking is enabled,
                return a ThinkingResponse containing both thinking and content.
            **kwargs: Additional parameters (max_tokens, temperature, top_p, etc.)

        Returns:
            Union[str, ThinkingResponse, Iterator[str]]: Response string,
                ThinkingResponse (if return_thinking=True), or iterator if streaming

        Examples:
            # Simple query
            response = llm.invoke(
                [{"role": "user", "content": "Hello!"}],
                "You are a helpful assistant."
            )

            # Streaming
            for token in llm.invoke(messages, system, streaming=True):
                print(token, end="", flush=True)

            # Structured output
            from pydantic import BaseModel
            class Answer(BaseModel):
                answer: str
                confidence: float

            response = llm.invoke(messages, system, schema=Answer)

            # With thinking mode
            response = llm.invoke(
                messages, system,
                use_thinking="high",
                return_thinking=True
            )
        """
        # Use thinking endpoint if requested
        if use_thinking:
            if images:
                logger.warning("Images not supported with thinking mode. Ignoring.")
            if schema:
                logger.warning("Schema not supported with thinking mode. Ignoring.")
            return self._invoke_with_thinking(
                messages=messages,
                system_prompt=system_prompt,
                streaming=streaming,
                model_name=model_name,
                use_thinking=use_thinking,
                return_thinking=return_thinking,
                **kwargs
            )
        
        # Build messages with system prompt
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Handle images for vision models
        if images:
            content = []
            for img in images:
                encoded = _encode_image(img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
                })
            content.append({
                "type": "text",
                "text": full_messages[-1]["content"]
            })
            full_messages[-1] = {
                "role": full_messages[-1]["role"],
                "content": content
            }
        
        # Build completion parameters
        completion_params = {
            "model": model_name or self.model_name,
            "messages": full_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        # Add optional parameters
        if "top_p" in kwargs:
            completion_params["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            completion_params["top_k"] = kwargs["top_k"]
        if "stop" in kwargs:
            completion_params["stop"] = kwargs["stop"]
        if "presence_penalty" in kwargs:
            completion_params["presence_penalty"] = kwargs["presence_penalty"]
        if "frequency_penalty" in kwargs:
            completion_params["frequency_penalty"] = kwargs["frequency_penalty"]
        if "seed" in kwargs:
            completion_params["seed"] = kwargs["seed"]
        
        # Add structured output support
        if schema:
            completion_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema()
                }
            }
        
        # Retry logic for model-not-loaded errors
        max_retries = kwargs.get('_model_reload_retries', DEFAULT_MODEL_RELOAD_RETRIES)
        wait_times = kwargs.get('_model_reload_wait_seconds', DEFAULT_MODEL_RELOAD_WAIT_SECONDS)
        
        last_error = None
        for attempt in range(max_retries):
            try:
                if streaming:
                    completion_params["stream"] = True
                    stream = self.client.chat.completions.create(**completion_params)
                    
                    def _gen() -> Iterator[str]:
                        for chunk in stream:
                            if chunk.choices and chunk.choices[0].delta.content:
                                yield chunk.choices[0].delta.content
                    
                    return _gen()
                else:
                    response = self.client.chat.completions.create(**completion_params)
                    return response.choices[0].message.content
                    
            except Exception as e:
                error_str = str(e).lower()
                last_error = e
                
                # Check if this is a model-not-loaded error
                is_model_error = (
                    'no model found' in error_str or
                    'model not found' in error_str or
                    'nomodelmatchingquery' in error_str or
                    'model has crashed' in error_str or
                    ('totalloadedmodels' in error_str and '0' in error_str) or
                    'could not find' in error_str
                )
                
                if is_model_error and attempt < max_retries - 1:
                    wait_time = wait_times[min(attempt, len(wait_times) - 1)]
                    logger.warning(
                        f"Model '{self.model_name}' not available (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    
                    # Try to ensure model is loaded
                    if not self.is_model_loaded():
                        logger.info(f"Attempting to load model {self.model_name}...")
                        self.load_model_explicit(timeout=60)
                    
                    continue
                else:
                    break
        
        # All retries failed
        raise RuntimeError(
            f"Error during LM Studio inference: {last_error}. "
            f"Check that model '{self.model_name}' is available."
        ) from last_error

    def close(self):
        """Clean up resources (no-op for HTTP-based client)."""
        pass

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.close()
