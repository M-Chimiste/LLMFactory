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

"""Anthropic inference providers."""

import os
from typing import List, Dict, Union, Optional, Iterator

from .base import InferenceModel, _encode_image


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
