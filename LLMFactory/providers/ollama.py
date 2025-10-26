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

"""Ollama inference providers."""

import os
import numpy as np
from typing import List, Dict, Union, Optional, Iterator
from pydantic import BaseModel

from .base import InferenceModel, _encode_image


class OllamaInference(InferenceModel):
    """Ollama Inference for Ollama's API."""
    def __init__(self,
                 model_name: str = "hermes3",
                 max_new_tokens: int = 4096,
                 temperature: float = 0.1,
                 url: str = None,
                 num_ctx: int = 131072,
                 request_timeout: Optional[float] = None):
        self.url = url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
        self.num_ctx = num_ctx
        self.request_timeout = request_timeout  # Timeout in seconds, None = no timeout
        super().__init__(model_name, max_new_tokens, temperature)

    def close(self):
        """Close the underlying HTTP client to release connections."""
        if hasattr(self, 'client') and self.client:
            if hasattr(self.client, '_client'):
                self.client._client.close()

    def __del__(self):
        """Ensure connections are closed when object is garbage collected."""
        self.close()

    def _get_provider(self) -> str:
        return "ollama"

    def _load_model(self):
        from ollama import Client
        # Create client with timeout if specified
        if self.request_timeout is not None:
            return Client(host=self.url, timeout=self.request_timeout)
        else:
            return Client(host=self.url)

    def invoke(self, messages: List[Dict[str, str]], system_prompt: str, *,
               streaming: bool = False,
               model_name: Optional[str] = None, num_ctx: Optional[int] = None,
               schema: Optional[BaseModel] = None,
               images: Optional[List[Union[str, bytes]]] = None) -> Union[str, Iterator[str]]:
        """
        Invokes the Ollama model to generate a response based on the provided messages and system prompt.

        This method orchestrates the interaction with the Ollama model, handling both streaming and non-streaming modes.
        It prepares the input messages, including the system prompt, and configures the model invocation options.
        Depending on the streaming flag, it either returns the full response string or an iterator yielding tokens.

        Args:
            messages (List[Dict[str, str]]): A list of dictionaries, each containing a 'role' and 'content' for the message.
            system_prompt (str): The system prompt to be included in the input.
            streaming (bool, optional): If True, returns an iterator over the response tokens. Defaults to False.
            model_name (Optional[str], optional): The name of the model to use. Defaults to the model name set in the constructor.
            num_ctx (Optional[int], optional): The number of context tokens to use. Defaults to the num_ctx set in the constructor.
            schema (Optional[BaseModel], optional): The schema to use for formatting the response. Defaults to None.
            images (Optional[List[Union[str, bytes]]], optional): List of image paths or bytes for multimodal inference. Defaults to None.

        Returns:
            Union[str, Iterator[str]]: The response from the model, either as a full string or an iterator over tokens.
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        if images:
            full_messages[-1]["images"] = [_encode_image(img) for img in images]

        options = {
            "num_predict": self.max_new_tokens,
            "temperature": self.temperature,
            "num_ctx": self.num_ctx
        }

        if streaming:
            if schema:
                stream = self.client.chat(
                    model=model_name or self.model_name,
                    messages=full_messages,
                    format=schema.model_json_schema(),
                    options=options,
                    stream=True
                )
            else:
                stream = self.client.chat(
                    model=model_name or self.model_name,
                    messages=full_messages,
                    options=options,
                    stream=True
                )

            def _gen() -> Iterator[str]:
                for chunk in stream:
                    content = chunk["message"]["content"]
                    if content:
                        yield content
            return _gen()
        else:
            if schema:
                response = self.client.chat(
                    model=model_name or self.model_name,
                    messages=full_messages,
                    format=schema.model_json_schema(),
                    options=options
                )
            else:
                response = self.client.chat(
                    model=model_name or self.model_name,
                    messages=full_messages,
                    options=options
                )
            return response['message']['content']


class OllamaEmbedInference(InferenceModel):
    """Ollama Embedding Inference for Ollama's embedding API."""
    def __init__(self,
                 model_name: str = "nomic-embed-text",
                 max_new_tokens: int = 4096,
                 temperature: float = 0.1,
                 trust_remote_code: bool = True,
                 url: str = None):
        """
        Initializes the Ollama Embedding Inference model.

        Args:
            model_name (str, optional): The name of the model to use for embedding generation. Defaults to "nomic-embed-text".
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 4096.
            temperature (float, optional): The temperature parameter for the model. Defaults to 0.1.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to True.
            url (str, optional): The URL of the Ollama API. If None, uses the OLLAMA_URL environment variable. Defaults to None.
        """
        self.url = url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
        super().__init__(model_name, max_new_tokens, temperature, trust_remote_code)

    def _get_provider(self) -> str:
        return "ollama-embed"

    def _load_model(self):
        from ollama import Client
        return Client(host=self.url)

    def invoke(self, text: Union[str, List[str]], *,
               streaming: bool = False,
               to_list: bool = False,
               normalize: bool = False, model_name: Optional[str] = None, **kwargs) -> Union[List, object]:
        """
        Invokes the Ollama embedding model to generate embeddings for the given text(s).

        This method takes in a single string or a list of strings as input, and returns the corresponding embeddings. It supports various options such as streaming, converting to list, normalizing, and specifying a custom model name.

        Args:
            text (Union[str, List[str]]): The input text(s) for which embeddings are to be generated.
            streaming (bool, optional): If True, the method will process the input text(s) in a streaming fashion. Defaults to False.
            to_list (bool, optional): If True, the method will convert the embeddings to a list format. Defaults to False.
            normalize (bool, optional): If True, the method will normalize the embeddings. Defaults to False.
            model_name (Optional[str], optional): The name of the model to use for embedding generation. Defaults to the model name specified during initialization.

        Returns:
            Union[List, object]: The generated embeddings. If to_list is True, returns a list of embeddings; otherwise, returns a single embedding if the input was a single string, or a list of embeddings if the input was a list of strings.
        """
        _ = streaming
        if isinstance(text, str):
            text = [text]

        # Generate embeddings for each text
        embeddings = []
        for single_text in text:
            response = self.client.embeddings(
                model=model_name or self.model_name,
                prompt=single_text
            )
            embeddings.append(response['embedding'])

        # Convert to numpy arrays for consistency with SentenceTransformer
        embeddings = [np.array(embedding) for embedding in embeddings]

        # Handle normalization if requested
        if normalize:
            embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]

        # Convert to list format if requested
        if to_list:
            embeddings = [embedding.tolist() for embedding in embeddings]

        # Return single embedding if only one text was provided
        if len(embeddings) == 1:
            return embeddings[0]
        return embeddings
