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

"""LLMFactory providers module - modular LLM inference providers."""

from .base import InferenceModel, _encode_image
from .ollama import OllamaInference, OllamaEmbedInference
from .lmstudio import LMStudioInference
from .anthropic import AnthropicInference, AnthropicBedrockInference
from .openai import OpenAIInference, CustomOAIInference
from .gemini import GeminiInference
from .embeddings import SentenceTransformerInference
from .llamacpp import LlamacppInference

__all__ = [
    # Base classes
    'InferenceModel',
    '_encode_image',

    # Ollama providers
    'OllamaInference',
    'OllamaEmbedInference',

    # LM Studio
    'LMStudioInference',

    # Anthropic providers
    'AnthropicInference',
    'AnthropicBedrockInference',

    # OpenAI providers
    'OpenAIInference',
    'CustomOAIInference',

    # Google Gemini
    'GeminiInference',

    # Embedding providers
    'SentenceTransformerInference',

    # Local inference
    'LlamacppInference',
]
