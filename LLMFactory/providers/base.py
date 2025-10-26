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

"""Base classes and utilities for inference models."""

import base64
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Iterator


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
