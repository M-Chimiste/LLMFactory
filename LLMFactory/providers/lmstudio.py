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

import os
from typing import List, Dict, Union, Optional, Iterator
from pydantic import BaseModel

from .base import InferenceModel, _encode_image


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
        """Initialize the LM Studio client."""
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

        # Configure the default client to use the specified host
        lms.configure_default_client(self.host)
        return lms

    def _get_or_load_model(self):
        """Get the model instance, loading it with config if not already loaded."""
        if self._model_instance is not None:
            return self._model_instance

        # Build load configuration
        config = {}
        if self.context_length is not None:
            config["context_length"] = self.context_length
        if self.gpu_offload is not None:
            if isinstance(self.gpu_offload, str):
                config["gpu_offload"] = self.gpu_offload  # "max" or "off"
            else:
                config["gpu"] = {"ratio": self.gpu_offload}

        # Load model with configuration (JIT loading)
        if config:
            self._model_instance = self.client.llm(self.model_name, config=config)
        else:
            self._model_instance = self.client.llm(self.model_name)

        return self._model_instance

    def invoke(self,
               messages: List[Dict[str, str]],
               system_prompt: str,
               *,
               streaming: bool = False,
               model_name: Optional[str] = None,
               schema: Optional[BaseModel] = None,
               images: Optional[List[Union[str, bytes]]] = None,
               **kwargs) -> Union[str, Iterator[str]]:
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
            **kwargs: Additional parameters:
                     - max_tokens: Override max_new_tokens
                     - temperature: Override temperature
                     - top_p, top_k, stop: Sampling parameters

        Returns:
            Union[str, Iterator[str]]: Full response string or iterator yielding tokens if streaming=True

        Examples:
            # Simple query
            response = llm.invoke(
                [{"role": "user", "content": "Hello!"}],
                "You are a helpful assistant."
            )

            # Streaming
            for token in llm.invoke(messages, system, streaming=True):
                print(token, end="", flush=True)

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
        # Get or load the model
        model = self._get_or_load_model()

        # Create Chat object for LM Studio SDK
        chat = self.client.Chat()

        # Add system prompt
        if system_prompt:
            chat.add_system_message(system_prompt)

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
                chat.add_system_message(content)
            elif role == "assistant":
                chat.add_assistant_message(content)
            else:  # user or any other role
                if is_last_user_msg and image_handles:
                    chat.add_user_message(content, images=image_handles)
                else:
                    chat.add_user_message(content)

        # Build generation parameters
        gen_params = {
            "max_tokens": kwargs.get("max_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        # Add structured output support if schema provided
        if schema:
            gen_params["response_format"] = schema

        # Add additional sampling parameters if provided
        for key in ["top_p", "top_k", "stop"]:
            if key in kwargs:
                gen_params[key] = kwargs[key]

        try:
            if streaming:
                # Return streaming iterator
                stream = model.respond_stream(chat, **gen_params)

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

                return _gen()
            else:
                # Return full response
                response = model.respond(chat, **gen_params)

                # Handle different possible response formats
                if hasattr(response, 'content'):
                    return response.content
                elif isinstance(response, dict) and 'content' in response:
                    return response['content']
                else:
                    return str(response)

        except Exception as e:
            raise RuntimeError(
                f"Error during LM Studio inference: {str(e)}. "
                f"Check that model '{self.model_name}' is available and context length is appropriate."
            ) from e

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
