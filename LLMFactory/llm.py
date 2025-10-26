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
import os
import base64
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Iterator
from pydantic import BaseModel
import numpy as np
import inspect

def _encode_image(image_data: Union[str, bytes]) -> str:
    """Convert image file path or bytes to base64 string."""
    if isinstance(image_data, bytes):
        return base64.b64encode(image_data).decode('utf-8')
    with open(image_data, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

class InferenceModel(ABC):
    """
    Abstract base class for all inference models.
    """
    def __init__(self,
                 model_name: str,
                 max_new_tokens: int = 4096,
                 temperature: float = 0.1,
                 trust_remote_code: bool = True):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.trust_remote_code = trust_remote_code
        self.provider = self._get_provider()
        self.client = self._load_model()

    @abstractmethod
    def _load_model(self):
        """Load and return the model client."""
        pass

    @abstractmethod
    def _get_provider(self) -> str:
        """Return the provider name as a string."""
        pass

    @abstractmethod
    def invoke(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        *,
        streaming: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """Generate a response.

        If *streaming* is True and the underlying provider offers token
        streaming, return an iterator that yields those tokens.  Providers
        that do not support streaming MUST silently ignore the flag and
        return the full response string as before."""
        pass


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


class AnthropicInference(InferenceModel):
    """Anthropic Inference for Anthropic's API."""
    def _get_provider(self) -> str:
        return "anthropic"

    def _load_model(self):
        from anthropic import Anthropic
        return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    
    def invoke(self, messages: List[Dict[str, str]], system_prompt: str, *,
               streaming: bool = False, model_name: Optional[str] = None,
               images: Optional[List[Union[str, bytes]]] = None) -> Union[str, Iterator[str]]:
        """
        Generates a response using the Anthropic model.

        This method orchestrates the interaction with the Anthropic API to generate a response based on the provided messages and system prompt. It can operate in either streaming or non-streaming mode, depending on the value of the `streaming` parameter. In streaming mode, it returns an iterator over the generated text chunks. In non-streaming mode, it returns the complete generated response as a string.

        Args:
            messages (List[Dict[str, str]]): A list of dictionaries, each containing 'role' and 'content' keys, representing the conversation history.
            system_prompt (str): The system prompt to prepend to the conversation history.
            model_name (Optional[str]): The name of the model to use for generation. If not provided, the default model is used.
            images (Optional[List[Union[str, bytes]]], optional): List of image paths or bytes for multimodal inference. Defaults to None.

        Returns:
            Union[str, Iterator[str]]: The generated response text or an iterator over the generated text chunks if streaming is enabled.
        """
        if images:
            messages = messages.copy()
            content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _encode_image(img)}} for img in images]
            content.append({"type": "text", "text": messages[-1]["content"]})
            messages[-1] = {"role": messages[-1]["role"], "content": content}
        if streaming:
            stream = self.client.messages.create(
                model=model_name or self.model_name,
                system=system_prompt,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                stream=True,
            )

            def _gen() -> Iterator[str]:
                for event in stream:
                    if getattr(event, "type", None) == "content_block_delta":
                        delta = event.delta.get("text", "")
                        if delta:
                            yield delta
            return _gen()
        else:
            response = self.client.messages.create(
                model=model_name or self.model_name,
                system=system_prompt,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            return response.content[0].text


class AnthropicBedrockInference(InferenceModel):
    """Anthropic Bedrock Inference for AWS Bedrock's API."""
    def __init__(self,
                 model_name: str = "anthropic.claude-3-sonnet-20240229-v1:0",
                 max_new_tokens: int = 4096,
                 temperature: float = 0.1,
                 trust_remote_code: bool = True,
                 aws_profile: Optional[str] = None,
                 region_name: Optional[str] = None):
        """
        Initialize Anthropic Bedrock Inference.

        Args:
            model_name (str): The Bedrock model ID. Defaults to Claude 3 Sonnet.
            max_new_tokens (int): Maximum tokens to generate. Defaults to 4096.
            temperature (float): Temperature for sampling. Defaults to 0.1.
            trust_remote_code (bool): Whether to trust remote code. Defaults to True.
            aws_profile (str, optional): AWS CLI profile name. Defaults to None.
            region_name (str, optional): AWS region. Defaults to us-east-1.
        """
        self.aws_profile = aws_profile or os.environ.get("AWS_PROFILE")
        self.region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        self._credentials = None
        super().__init__(model_name, max_new_tokens, temperature, trust_remote_code)

    def _get_provider(self) -> str:
        return "anthropic-bedrock"

    def _get_credentials(self):
        """Get AWS credentials from profile, environment, or boto3 session."""
        import boto3
        from botocore.exceptions import BotoCoreError, NoCredentialsError, ProfileNotFound

        # Try 1: AWS profile
        if self.aws_profile:
            try:
                session = boto3.Session(profile_name=self.aws_profile, region_name=self.region_name)
                credentials = session.get_credentials()
                if credentials:
                    return {
                        'aws_access_key_id': credentials.access_key,
                        'aws_secret_access_key': credentials.secret_key,
                        'aws_session_token': credentials.token,
                        'region_name': self.region_name
                    }
            except ProfileNotFound:
                pass

        # Try 2: Environment variables
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_session_token = os.environ.get("AWS_SESSION_TOKEN")

        if aws_access_key and aws_secret_key:
            return {
                'aws_access_key_id': aws_access_key,
                'aws_secret_access_key': aws_secret_key,
                'aws_session_token': aws_session_token,
                'region_name': self.region_name
            }

        # Try 3: boto3 default session (EC2 instance role, etc.)
        try:
            session = boto3.Session(region_name=self.region_name)
            credentials = session.get_credentials()
            if credentials:
                return {
                    'aws_access_key_id': credentials.access_key,
                    'aws_secret_access_key': credentials.secret_key,
                    'aws_session_token': credentials.token,
                    'region_name': self.region_name
                }
        except (BotoCoreError, NoCredentialsError):
            pass

        raise ValueError(
            "No valid AWS credentials found. Please provide credentials via:\n"
            "1. AWS CLI profile (aws_profile parameter or AWS_PROFILE env var)\n"
            "2. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)\n"
            "3. EC2 instance role or other boto3 credential sources"
        )

    def _load_model(self):
        from anthropic import AnthropicBedrock

        self._credentials = self._get_credentials()

        return AnthropicBedrock(
            aws_access_key=self._credentials['aws_access_key_id'],
            aws_secret_key=self._credentials['aws_secret_access_key'],
            aws_session_token=self._credentials.get('aws_session_token'),
            aws_region=self._credentials['region_name']
        )

    def invoke(self, messages: List[Dict[str, str]], system_prompt: str, *,
               streaming: bool = False, model_name: Optional[str] = None,
               images: Optional[List[Union[str, bytes]]] = None) -> Union[str, Iterator[str]]:
        """
        Generates a response using the Anthropic Bedrock model.

        This method orchestrates the interaction with the Anthropic Bedrock API to generate a response based on the provided messages and system prompt. It can operate in either streaming or non-streaming mode, depending on the value of the `streaming` parameter. In streaming mode, it returns an iterator over the generated text chunks. In non-streaming mode, it returns the complete generated response as a string.

        Args:
            messages (List[Dict[str, str]]): A list of dictionaries, each containing 'role' and 'content' keys, representing the conversation history.
            system_prompt (str): The system prompt to prepend to the conversation history.
            model_name (Optional[str]): The name of the model to use for generation. If not provided, the default model is used.
            images (Optional[List[Union[str, bytes]]], optional): List of image paths or bytes for multimodal inference. Defaults to None.

        Returns:
            Union[str, Iterator[str]]: The generated response text or an iterator over the generated text chunks if streaming is enabled.
        """
        if images:
            messages = messages.copy()
            content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _encode_image(img)}} for img in images]
            content.append({"type": "text", "text": messages[-1]["content"]})
            messages[-1] = {"role": messages[-1]["role"], "content": content}

        if streaming:
            stream = self.client.messages.create(
                model=model_name or self.model_name,
                system=system_prompt,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                stream=True,
            )

            def _gen() -> Iterator[str]:
                for event in stream:
                    if getattr(event, "type", None) == "content_block_delta":
                        delta = event.delta.get("text", "")
                        if delta:
                            yield delta
            return _gen()
        else:
            response = self.client.messages.create(
                model=model_name or self.model_name,
                system=system_prompt,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            return response.content[0].text


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


class SentenceTransformerInference(InferenceModel):
    def __init__(self,
                 model_name: str = "Alibaba-NLP/gte-large-en-v1.5",
                 remote_code: bool = True,
                 device: Optional[str] = None):
        """
        Initializes a SentenceTransformerInference model instance.

        This constructor sets up the SentenceTransformer model for inference tasks. It allows for customization of the model name, remote code execution, and device selection.

        Args:
            model_name (str, optional): The name of the SentenceTransformer model to use. Defaults to "Alibaba-NLP/gte-large-en-v1.5".
            remote_code (bool, optional): Whether to allow remote code execution. Defaults to True.
            device (Optional[str], optional): The device to use for model computations. Defaults to None, which automatically selects the best available device.

        Raises:
            ValueError: If an invalid device is specified.
        """
        import torch
        self.remote_code = remote_code
        if not device:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        super().__init__(model_name)

    def _get_provider(self) -> str:
        return "sentence-transformer"

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.model_name, trust_remote_code=self.remote_code, device=self.device)
    
    def invoke(self,
               text: Union[str, List[str]],
               *,
               streaming: bool = False,
               to_list: bool = False,
               normalize: bool = False,
               batch_size: Optional[int] = None,
               show_progress_bar: Optional[bool] = None,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               **kwargs) -> Union[List, object]:
        """
        Invokes the SentenceTransformer model to generate embeddings for the given text(s).

        This method processes the input text(s) and returns their corresponding embeddings. It supports
        various options for customization, such as streaming, converting to list, normalization, and more.

        Args:
            text (Union[str, List[str]]): The input text or list of texts to generate embeddings for.
            streaming (bool, optional): If True, processes the input text in a streaming fashion. Defaults to False.
            to_list (bool, optional): If True, converts the output embeddings to a list. Defaults to False.
            normalize (bool, optional): If True, normalizes the embeddings. Defaults to False.
            batch_size (Optional[int], optional): The batch size to use for encoding. Defaults to None.
            show_progress_bar (Optional[bool], optional): If True, shows a progress bar during encoding. Defaults to None.
            convert_to_numpy (bool, optional): If True, converts the embeddings to numpy arrays. Defaults to True.
            convert_to_tensor (bool, optional): If True, converts the embeddings to tensors. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the encoding function.

        Returns:
            Union[List, object]: The generated embeddings. If `to_list` is True, returns a list of embeddings. Otherwise, returns the embeddings as is.
        """
        _ = streaming
        is_string_input = isinstance(text, str)
        if is_string_input:
            text = [text]
        
        # Build encode parameters
        encode_params = {
            'sentences': text,
            'normalize_embeddings': normalize,
            'convert_to_numpy': convert_to_numpy,
            'convert_to_tensor': convert_to_tensor,
            **kwargs
        }
        
        # Add optional parameters if provided
        if batch_size is not None:
            encode_params['batch_size'] = batch_size
        if show_progress_bar is not None:
            encode_params['show_progress_bar'] = show_progress_bar
            
        embeddings = self.client.encode(**encode_params)
        
        if to_list and convert_to_numpy:
            embeddings = [embedding.tolist() for embedding in embeddings]
        
        # Only return single embedding if input was a single string, not a list
        if is_string_input and len(embeddings) == 1:
            return embeddings[0]
        return embeddings


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


class LLMModelFactory:
    """
    Factory class for creating inference model instances.
    """
    _models = {
        'ollama': OllamaInference,
        'lmstudio': LMStudioInference,
        'anthropic': AnthropicInference,
        'anthropic-bedrock': AnthropicBedrockInference,
        'openai': OpenAIInference,
        'gemini': GeminiInference,
        'custom_oai': CustomOAIInference,
        'sentence-transformer': SentenceTransformerInference,
        'ollama-embed': OllamaEmbedInference,
        'llamacpp': LlamacppInference
    }

    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> InferenceModel:
        """
        Create and return an instance of the specified inference model.
        This method inspects the model's constructor and only passes arguments
        that it can accept, preventing TypeErrors for unexpected keywords.
        """
        model_class = cls._models.get(model_type.lower())
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")

        # Inspect the constructor signature to filter out unexpected arguments
        sig = inspect.signature(model_class.__init__)
        allowed_args = {p.name for p in sig.parameters.values()}
        
        # Check if the constructor accepts arbitrary keyword arguments (**kwargs)
        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

        if has_kwargs:
            # If it does, no need to filter.
            filtered_kwargs = kwargs
        else:
            # Otherwise, filter to only include arguments present in the signature.
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_args}

        return model_class(**filtered_kwargs)
        
        
