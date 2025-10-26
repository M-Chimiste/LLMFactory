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

"""llama.cpp inference provider."""

from typing import List, Dict, Union, Optional, Iterator
from pydantic import BaseModel

from .base import InferenceModel, _encode_image


class LlamacppInference(InferenceModel):
    """Llamacpp Inference for local GGUF models using llama-cpp-python."""
    def __init__(self,
                 model_name: str,
                 max_new_tokens: int = 4096,
                 temperature: float = 0.1,
                 trust_remote_code: bool = True,
                 num_ctx: int = 131072,
                 n_gpu_layers: int = -1,
                 verbose: bool = False,
                 **kwargs):
        """
        Initializes the LlamacppInference model with the specified parameters.

        Args:
            model_name (str): The name of the model to use for inference.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 4096.
            temperature (float, optional): The temperature parameter for sampling. Defaults to 0.1.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to True.
            num_ctx (int, optional): The number of context tokens to use. Defaults to 131072.
            n_gpu_layers (int, optional): The number of GPU layers to use. Defaults to -1.
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the model.

        Raises:
            ValueError: If the model_name is not specified.
        """
        self.num_ctx = num_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.model_kwargs = kwargs
        super().__init__(model_name, max_new_tokens, temperature, trust_remote_code)

    def _get_provider(self) -> str:
        return "llamacpp"

    def _load_model(self):
        from llama_cpp import Llama
        return Llama(
            model_path=self.model_name,
            n_ctx=self.num_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
            **self.model_kwargs
        )

    def invoke(self, messages: List[Dict[str, str]], system_prompt: str, *,
               streaming: bool = False,
               max_tokens: Optional[int] = None,
               temperature: Optional[float] = None,
               schema: Optional[BaseModel] = None,
               images: Optional[List[Union[str, bytes]]] = None,
               **kwargs) -> Union[str, Iterator[str]]:
        """
        Initiates the inference process for generating text based on input messages and a system prompt.

        This method orchestrates the text generation process by preparing the input messages, system prompt, and additional parameters for the model. It supports both streaming and non-streaming modes of operation. In streaming mode, it yields chunks of generated text as they become available. In non-streaming mode, it returns the complete generated text.

        Args:
            messages (List[Dict[str, str]]): A list of dictionaries, where each dictionary contains a 'role' and 'content'. The 'role' specifies the type of message (e.g., 'system', 'user'), and 'content' is the text of the message.
            system_prompt (str): A string that provides context or instructions for the model to generate text.
            streaming (bool, optional): If True, the method operates in streaming mode, yielding chunks of generated text. Defaults to False.
            max_tokens (Optional[int], optional): The maximum number of tokens to generate. Defaults to the model's default max_new_tokens.
            temperature (Optional[float], optional): The temperature parameter for sampling. Defaults to the model's default temperature.
            schema (Optional[BaseModel], optional): A Pydantic model schema for structured output. Defaults to None.
            images (Optional[List[Union[str, bytes]]], optional): List of image paths or bytes for multimodal inference. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the model for generation.

        Returns:
            Union[str, Iterator[str]]: In non-streaming mode, returns the complete generated text as a string. In streaming mode, returns an iterator that yields chunks of generated text as strings.
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        if images:
            content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(img)}"}} for img in images]
            content.append({"type": "text", "text": full_messages[-1]["content"]})
            full_messages[-1] = {"role": full_messages[-1]["role"], "content": content}

        # Prepare generation parameters
        generation_params = {
            "messages": full_messages,
            "max_tokens": max_tokens or self.max_new_tokens,
            "temperature": temperature or self.temperature,
            **kwargs
        }

        # Add schema support for structured output
        if schema:
            generation_params["response_format"] = {
                "type": "json_object",
                "schema": schema.model_json_schema()
            }

        if streaming:
            stream = self.client.create_chat_completion(**generation_params, stream=True)

            def _gen() -> Iterator[str]:
                for chunk in stream:
                    delta = chunk["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
            return _gen()
        else:
            response = self.client.create_chat_completion(**generation_params)
            return response['choices'][0]['message']['content']
