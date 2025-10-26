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

"""
LLMFactory - A unified interface for various LLM providers.

This module provides a factory pattern for creating LLM inference instances
from various providers. All provider implementations are now organized in
the 'providers' submodule for better maintainability and modularity.
"""

import inspect
from typing import Dict, Type

# Import all providers from the providers module
from .providers import (
    InferenceModel,
    _encode_image,
    OllamaInference,
    OllamaEmbedInference,
    LMStudioInference,
    AnthropicInference,
    AnthropicBedrockInference,
    OpenAIInference,
    CustomOAIInference,
    GeminiInference,
    SentenceTransformerInference,
    LlamacppInference,
)


class LLMModelFactory:
    """
    Factory class for creating inference model instances.
    """
    _models: Dict[str, Type[InferenceModel]] = {
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


# Export all public classes for backward compatibility
__all__ = [
    # Factory
    'LLMModelFactory',

    # Base class and utilities
    'InferenceModel',
    '_encode_image',

    # All provider classes
    'OllamaInference',
    'OllamaEmbedInference',
    'LMStudioInference',
    'AnthropicInference',
    'AnthropicBedrockInference',
    'OpenAIInference',
    'CustomOAIInference',
    'GeminiInference',
    'SentenceTransformerInference',
    'LlamacppInference',
]
