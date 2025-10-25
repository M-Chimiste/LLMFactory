"""Tests for LlamacppInference."""
import pytest
from unittest.mock import Mock, patch
from LLMFactory.llm import LlamacppInference


@patch('llama_cpp.Llama')
def test_llamacpp_init(mock_llama_class, tmp_path):
    """Test LlamacppInference initialization."""
    model_file = tmp_path / "model.gguf"
    model_file.write_text("dummy")

    mock_llama_class.return_value = Mock()

    model = LlamacppInference(
        model_name=str(model_file),
        num_ctx=4096,
        n_gpu_layers=35
    )

    assert model.model_name == str(model_file)
    assert model.num_ctx == 4096
    assert model.n_gpu_layers == 35
    assert model.provider == "llamacpp"


@patch('llama_cpp.Llama')
def test_llamacpp_invoke_basic(mock_llama_class, tmp_path, sample_messages, sample_system_prompt):
    """Test basic invoke without streaming."""
    model_file = tmp_path / "model.gguf"
    model_file.write_text("dummy")

    mock_client = Mock()
    mock_client.create_chat_completion.return_value = {
        'choices': [{'message': {'content': 'Llama response'}}]
    }
    mock_llama_class.return_value = mock_client

    model = LlamacppInference(model_name=str(model_file))
    response = model.invoke(sample_messages, sample_system_prompt)

    assert response == 'Llama response'
    mock_client.create_chat_completion.assert_called_once()


@patch('llama_cpp.Llama')
def test_llamacpp_invoke_streaming(mock_llama_class, tmp_path, sample_messages, sample_system_prompt):
    """Test invoke with streaming."""
    model_file = tmp_path / "model.gguf"
    model_file.write_text("dummy")

    mock_client = Mock()
    mock_chunks = [
        {'choices': [{'delta': {'content': 'Hello '}}]},
        {'choices': [{'delta': {'content': 'world'}}]},
        {'choices': [{'delta': {}}]},
    ]
    mock_client.create_chat_completion.return_value = iter(mock_chunks)
    mock_llama_class.return_value = mock_client

    model = LlamacppInference(model_name=str(model_file))
    response = model.invoke(sample_messages, sample_system_prompt, streaming=True)

    result = ''.join(list(response))
    assert result == 'Hello world'


@patch('llama_cpp.Llama')
def test_llamacpp_invoke_with_images(mock_llama_class, tmp_path, sample_messages, sample_system_prompt, sample_image_file):
    """Test invoke with images (multimodal)."""
    model_file = tmp_path / "model.gguf"
    model_file.write_text("dummy")

    mock_client = Mock()
    mock_client.create_chat_completion.return_value = {
        'choices': [{'message': {'content': 'I see an image'}}]
    }
    mock_llama_class.return_value = mock_client

    model = LlamacppInference(model_name=str(model_file))
    response = model.invoke(
        sample_messages,
        sample_system_prompt,
        images=[sample_image_file]
    )

    assert response == 'I see an image'
    call_args = mock_client.create_chat_completion.call_args
    messages = call_args[1]['messages']

    # Check that the last message has content array with image_url
    assert isinstance(messages[-1]['content'], list)
    assert any(item['type'] == 'image_url' for item in messages[-1]['content'])


@patch('llama_cpp.Llama')
def test_llamacpp_invoke_with_schema(mock_llama_class, tmp_path, sample_messages, sample_system_prompt):
    """Test invoke with schema."""
    from pydantic import BaseModel

    class TestSchema(BaseModel):
        name: str
        age: int

    model_file = tmp_path / "model.gguf"
    model_file.write_text("dummy")

    mock_client = Mock()
    mock_client.create_chat_completion.return_value = {
        'choices': [{'message': {'content': '{"name": "Alice", "age": 25}'}}]
    }
    mock_llama_class.return_value = mock_client

    model = LlamacppInference(model_name=str(model_file))
    response = model.invoke(sample_messages, sample_system_prompt, schema=TestSchema)

    assert '{"name": "Alice", "age": 25}' in response
    call_args = mock_client.create_chat_completion.call_args
    assert 'response_format' in call_args[1]


@patch('llama_cpp.Llama')
def test_llamacpp_custom_params(mock_llama_class, tmp_path, sample_messages, sample_system_prompt):
    """Test invoke with custom max_tokens and temperature."""
    model_file = tmp_path / "model.gguf"
    model_file.write_text("dummy")

    mock_client = Mock()
    mock_client.create_chat_completion.return_value = {
        'choices': [{'message': {'content': 'Response'}}]
    }
    mock_llama_class.return_value = mock_client

    model = LlamacppInference(model_name=str(model_file))
    response = model.invoke(
        sample_messages,
        sample_system_prompt,
        max_tokens=1024,
        temperature=0.8
    )

    call_args = mock_client.create_chat_completion.call_args
    assert call_args[1]['max_tokens'] == 1024
    assert call_args[1]['temperature'] == 0.8