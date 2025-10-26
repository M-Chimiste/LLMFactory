"""Tests for LLMModelFactory."""
import pytest
from LLMFactory import LLMModelFactory
from LLMFactory.llm import (
    OllamaInference, LMStudioInference, AnthropicInference, OpenAIInference,
    GeminiInference, CustomOAIInference, SentenceTransformerInference,
    OllamaEmbedInference, LlamacppInference
)


def test_factory_create_ollama():
    """Test creating Ollama model."""
    model = LLMModelFactory.create_model('ollama', model_name='test-model')
    assert isinstance(model, OllamaInference)
    assert model.model_name == 'test-model'


def test_factory_create_lmstudio(mock_env_vars):
    """Test creating LMStudio model."""
    # Mock lmstudio module
    import sys
    from unittest.mock import Mock
    mock_lms = Mock()
    mock_lms.Client.is_valid_api_host.return_value = True
    sys.modules['lmstudio'] = mock_lms

    try:
        model = LLMModelFactory.create_model('lmstudio', model_name='qwen2.5-7b-instruct')
        assert isinstance(model, LMStudioInference)
        assert model.model_name == 'qwen2.5-7b-instruct'
    finally:
        # Cleanup
        if 'lmstudio' in sys.modules:
            del sys.modules['lmstudio']


def test_factory_create_anthropic(mock_env_vars, monkeypatch):
    """Test creating Anthropic model."""
    monkeypatch.setattr('anthropic.Anthropic', lambda **kwargs: None)
    model = LLMModelFactory.create_model('anthropic', model_name='claude-3-opus')
    assert isinstance(model, AnthropicInference)
    assert model.model_name == 'claude-3-opus'


def test_factory_create_openai(mock_env_vars, monkeypatch):
    """Test creating OpenAI model."""
    monkeypatch.setattr('openai.OpenAI', lambda **kwargs: None)
    model = LLMModelFactory.create_model('openai', model_name='gpt-4')
    assert isinstance(model, OpenAIInference)
    assert model.model_name == 'gpt-4'


def test_factory_create_gemini(mock_env_vars, monkeypatch):
    """Test creating Gemini model."""
    # Mock the genai configure function to accept api_key parameter
    monkeypatch.setattr('google.generativeai.configure', lambda api_key=None, **kwargs: None)
    
    model = LLMModelFactory.create_model('gemini', model_name='gemini-pro')
    assert isinstance(model, GeminiInference)
    assert model.model_name == 'gemini-pro'


def test_factory_create_custom_oai(mock_env_vars, monkeypatch):
    """Test creating custom OpenAI-compatible model."""
    monkeypatch.setattr('openai.OpenAI', lambda **kwargs: None)
    model = LLMModelFactory.create_model('custom_oai', model_name='custom-model')
    assert isinstance(model, CustomOAIInference)
    assert model.model_name == 'custom-model'


def test_factory_create_llamacpp(tmp_path, monkeypatch):
    """Test creating Llamacpp model."""
    # Create a dummy model file
    model_file = tmp_path / "model.gguf"
    model_file.write_text("dummy")

    monkeypatch.setattr('llama_cpp.Llama', lambda **kwargs: None)
    model = LLMModelFactory.create_model('llamacpp', model_name=str(model_file))
    assert isinstance(model, LlamacppInference)
    assert model.model_name == str(model_file)


def test_factory_unknown_model_type():
    """Test creating unknown model type raises error."""
    with pytest.raises(ValueError, match="Unknown model type"):
        LLMModelFactory.create_model('unknown_type')


def test_factory_filters_kwargs(mock_env_vars, monkeypatch):
    """Test factory filters out invalid kwargs."""
    monkeypatch.setattr('openai.OpenAI', lambda **kwargs: None)
    # OpenAI doesn't accept 'num_ctx', but Ollama does
    model = LLMModelFactory.create_model(
        'openai',
        model_name='gpt-4',
        num_ctx=4096,  # This should be filtered out
        temperature=0.5
    )
    assert isinstance(model, OpenAIInference)
    assert model.temperature == 0.5


def test_factory_preserves_kwargs_with_var_keyword(tmp_path, monkeypatch):
    """Test factory preserves all kwargs for models with **kwargs."""
    model_file = tmp_path / "model.gguf"
    model_file.write_text("dummy")

    monkeypatch.setattr('llama_cpp.Llama', lambda **kwargs: None)
    # LlamacppInference accepts **kwargs
    model = LLMModelFactory.create_model(
        'llamacpp',
        model_name=str(model_file),
        custom_param='value'  # This should be preserved
    )
    assert isinstance(model, LlamacppInference)