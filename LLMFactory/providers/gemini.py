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

"""Google Gemini inference provider."""

import os
from typing import List, Dict, Union, Optional, Iterator
from pydantic import BaseModel

from .base import InferenceModel


class GeminiInference(InferenceModel):
    """Gemini Inference for Google's Gemini API."""
    def __init__(self, model_name: str = "gemini-1.5-flash",
                 max_new_tokens: int = 4096,
                 temperature: float = 0.1):
        """
        Initializes the GeminiInference model with specified parameters.

        Args:
            model_name (str, optional): The name of the Gemini model to use for inference. Defaults to "gemini-1.5-flash".
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 4096.
            temperature (float, optional): The temperature parameter for the model. Defaults to 0.1.

        Raises:
            ValueError: If the model_name is not recognized or if max_new_tokens or temperature are invalid.
        """
        self.safety = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        super().__init__(model_name, max_new_tokens, temperature)

    def _get_provider(self) -> str:
        return "gemini"

    def _load_model(self):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        return genai

    def invoke(self,
               messages: List[Dict[str, str]],
               system_prompt: str, *,
               streaming: bool = False,
               model_name: Optional[str] = None,
               schema: Optional[BaseModel] = None,
               images: Optional[List[Union[str, bytes]]] = None,
               **kwargs) -> Union[str, Iterator[str]]:
        """
        Initiates the Gemini Inference process for generating text based on input messages and a system prompt.

        This method orchestrates the Gemini Inference process, which involves preparing input messages, setting up the model configuration, and invoking the model to generate text. It supports both streaming and non-streaming modes of operation.

        Args:
            messages (List[Dict[str, str]]): A list of dictionaries, where each dictionary contains a 'role' and a 'content'. The 'role' specifies the type of message (e.g., user or model), and the 'content' is the actual message.
            system_prompt (str): A string that serves as the initial prompt for the model to generate text.
            streaming (bool, optional): A boolean indicating whether to use streaming mode. Defaults to False.
            model_name (Optional[str], optional): The name of the model to use for inference. Defaults to None, which uses the default model name set during initialization.
            schema (Optional[BaseModel], optional): A Pydantic model schema for structured JSON output. Defaults to None.
            images (Optional[List[Union[str, bytes]]], optional): List of image paths or bytes for multimodal inference. Defaults to None.
            **kwargs: Additional keyword arguments that can be passed to the model for configuration.

        Returns:
            Union[str, Iterator[str]]: The generated text. If streaming is True, returns an iterator over the generated text. Otherwise, returns the complete generated text as a string.
        """
        gemini_messages = [{"role": "user", "parts": [system_prompt]}]
        for message in messages:
            role = "model" if message["role"] == "assistant" else "user"
            gemini_messages.append({"role": role, "parts": [message["content"]]})

        if images:
            from PIL import Image
            import io
            parts = []
            for img in images:
                if isinstance(img, bytes):
                    parts.append(Image.open(io.BytesIO(img)))
                else:
                    parts.append(Image.open(img))
            parts.append(gemini_messages[-1]["parts"][0])
            gemini_messages[-1]["parts"] = parts

        if streaming:
            model = self.client.GenerativeModel(model_name=model_name or self.model_name)

            # Prepare generation config
            generation_config_params = {
                "max_output_tokens": self.max_new_tokens,
                "temperature": self.temperature
            }

            # Add schema support for structured output
            if schema:
                generation_config_params["response_mime_type"] = "application/json"
                generation_config_params["response_schema"] = schema.model_json_schema()

            stream = model.generate_content(
                gemini_messages,
                safety_settings=self.safety,
                generation_config=self.client.types.GenerationConfig(**generation_config_params),
                stream=True
            )

            def _gen() -> Iterator[str]:
                for chunk in stream:
                    text = getattr(chunk, "text", None)
                    if text:
                        yield text
            return _gen()
        else:
            model = self.client.GenerativeModel(model_name=model_name or self.model_name)

            # Prepare generation config
            generation_config_params = {
                "max_output_tokens": self.max_new_tokens,
                "temperature": self.temperature
            }

            # Add schema support for structured output
            if schema:
                generation_config_params["response_mime_type"] = "application/json"
                generation_config_params["response_schema"] = schema.model_json_schema()

            response = model.generate_content(
                gemini_messages,
                safety_settings=self.safety,
                generation_config=self.client.types.GenerationConfig(**generation_config_params)
            )
            return response.text
