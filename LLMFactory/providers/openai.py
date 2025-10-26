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

"""OpenAI and OpenAI-compatible inference providers."""

import os
from typing import List, Dict, Union, Optional, Iterator
from pydantic import BaseModel

from .base import InferenceModel, _encode_image


class OpenAIInference(InferenceModel):
    """OpenAI Inference for OpenAI's API."""
    def _get_provider(self) -> str:
        return "openai"

    def _load_model(self):
        from openai import OpenAI
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def invoke(self, messages: List[Dict[str, str]], system_prompt: str, *,
               streaming: bool = False, model_name: Optional[str] = None,
               schema: Optional[BaseModel] = None,
               images: Optional[List[Union[str, bytes]]] = None) -> Union[str, Iterator[str]]:
        """
        Invokes the OpenAI model to generate a response based on the provided messages and system prompt.

        This method orchestrates the interaction with the OpenAI API to generate a response. It can operate in either streaming or non-streaming mode, depending on the value of the `streaming` parameter. In streaming mode, it returns an iterator over the generated text chunks. In non-streaming mode, it returns the complete generated response as a string.

        Args:
            messages (List[Dict[str, str]]): A list of dictionaries, each containing 'role' and 'content' keys, representing the conversation history.
            system_prompt (str): The system prompt to prepend to the conversation history.
            streaming (bool, optional): If True, the method returns an iterator over the generated text chunks. Defaults to False.
            model_name (Optional[str], optional): The name of the model to use for generation. If not provided, the default model is used.
            schema (Optional[BaseModel], optional): A Pydantic model schema for structured JSON output. Defaults to None.
            images (Optional[List[Union[str, bytes]]], optional): List of image paths or bytes for multimodal inference. Defaults to None.

        Returns:
            Union[str, Iterator[str]]: The generated response as a string or an iterator over the generated text chunks, depending on the `streaming` parameter.
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        if images:
            content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(img)}"}} for img in images]
            content.append({"type": "text", "text": full_messages[-1]["content"]})
            full_messages[-1] = {"role": full_messages[-1]["role"], "content": content}

        # Prepare completion parameters
        completion_params = {
            "model": model_name or self.model_name,
            "messages": full_messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

        # Add schema support for structured output
        if schema:
            completion_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema()
                }
            }

        if streaming:
            completion_params["stream"] = True
            stream = self.client.chat.completions.create(**completion_params)

            def _gen() -> Iterator[str]:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
            return _gen()
        else:
            response = self.client.chat.completions.create(**completion_params)
            return response.choices[0].message.content


class CustomOAIInference(InferenceModel):
    """Custom OpenAI-compatible Inference."""
    def __init__(self,
                 model_name: str,
                 max_new_tokens: int = 4096,
                 temperature: float = 0.1,
                 trust_remote_code: bool = True,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        """
        Initializes a CustomOAIInference instance with the specified model name, maximum new tokens, temperature, base URL, and API key.

        Args:
            model_name (str): The name of the model to use for inference.
            max_new_tokens (int, optional): The maximum number of tokens to generate in a single response. Defaults to 4096.
            temperature (float, optional): The temperature parameter for sampling. Defaults to 0.1.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to True.
            base_url (Optional[str], optional): The base URL for the custom OAI-compatible server. Defaults to None.
            api_key (Optional[str], optional): The API key for the custom OAI-compatible server. Defaults to None.

        Raises:
            ValueError: If the base_url or CUSTOM_OAI_BASE_URL environment variable is not set.
        """
        self.base_url = base_url or os.environ.get("CUSTOM_OAI_BASE_URL")
        self.api_key = api_key or os.environ.get("CUSTOM_OAI_API_KEY")
        if not self.base_url:
            raise ValueError("CUSTOM_OAI_BASE_URL environment variable or base_url parameter must be set for CustomOAIInference.")
        # API key can be optional for some self-hosted OAI compatible servers
        super().__init__(model_name, max_new_tokens, temperature, trust_remote_code)

    def _get_provider(self) -> str:
        return "custom-oai"

    def _load_model(self):
        from openai import OpenAI
        return OpenAI(base_url=self.base_url, api_key=self.api_key)

    def invoke(self,
               messages: List[Dict[str, str]],
               system_prompt: str, *,
               streaming: bool = False,
               model_name: Optional[str] = None,
               schema: Optional[BaseModel] = None,
               images: Optional[List[Union[str, bytes]]] = None,
               **kwargs) -> Union[str, Iterator[str]]:
        """
        Invokes the model to generate text based on the provided messages and system prompt.

        This method orchestrates the interaction with the model, preparing the input data and
        handling the response. It supports both streaming and non-streaming modes, allowing for
        flexible usage depending on the application's requirements.

        Args:
            messages (List[Dict[str, str]]): A list of dictionaries, where each dictionary contains
                'role' and 'content' keys. 'role' specifies the role of the message (e.g., 'user' or
                'system'), and 'content' is the actual message content.
            system_prompt (str): The system prompt to be used as the initial message.
            streaming (bool, optional): If True, the method will return an iterator over the
                generated text, allowing for streaming of the output. Defaults to False.
            model_name (Optional[str], optional): The name of the model to use for generation. If
                not provided, the default model name set during initialization will be used.
            schema (Optional[BaseModel], optional): A Pydantic model schema for structured JSON output. Defaults to None.
            images (Optional[List[Union[str, bytes]]], optional): List of image paths or bytes for multimodal inference. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the model's generation method.

        Returns:
            Union[str, Iterator[str]]: The generated text or an iterator over the generated text,
                depending on the streaming mode.
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        if images:
            content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(img)}"}} for img in images]
            content.append({"type": "text", "text": full_messages[-1]["content"]})
            full_messages[-1] = {"role": full_messages[-1]["role"], "content": content}

        # Prepare completion parameters
        completion_params = {
            "model": model_name or self.model_name,
            "messages": full_messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            **kwargs
        }

        # Add schema support for structured output
        if schema:
            completion_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema()
                }
            }

        if streaming:
            completion_params["stream"] = True
            stream = self.client.chat.completions.create(**completion_params)

            def _gen() -> Iterator[str]:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
            return _gen()
        else:
            response = self.client.chat.completions.create(**completion_params)
            return response.choices[0].message.content
