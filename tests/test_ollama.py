"""Tests for OllamaInference."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from LLMFactory.llm import OllamaInference


@patch('ollama.Client')
def test_ollama_init_default(mock_client_class):
    """Test OllamaInference initialization with defaults."""
    mock_client_class.return_value = Mock()
    model = OllamaInference()

    assert model.model_name == "hermes3"
    assert model.max_new_tokens == 4096
    assert model.temperature == 0.1
    assert model.url == "http://127.0.0.1:11434"
    assert model.provider == "ollama"


@patch('ollama.Client')
def test_ollama_init_custom(mock_client_class):
    """Test OllamaInference initialization with custom parameters."""
    mock_client_class.return_value = Mock()
    model = OllamaInference(
        model_name="llama2",
        max_new_tokens=2048,
        temperature=0.7,
        url="http://custom:8080",
        num_ctx=8192
    )

    assert model.model_name == "llama2"
    assert model.max_new_tokens == 2048
    assert model.temperature == 0.7
    assert model.url == "http://custom:8080"
    assert model.num_ctx == 8192


@patch('ollama.Client')
def test_ollama_invoke_basic(mock_client_class, sample_messages, sample_system_prompt):
    """Test basic invoke without streaming."""
    mock_client = Mock()
    mock_client.chat.return_value = {'message': {'content': 'Test response'}}
    mock_client_class.return_value = mock_client

    model = OllamaInference()
    response = model.invoke(sample_messages, sample_system_prompt)

    assert response == 'Test response'
    mock_client.chat.assert_called_once()
    call_args = mock_client.chat.call_args
    assert call_args[1]['model'] == 'hermes3'
    assert len(call_args[1]['messages']) == 4  # system + 3 messages


@patch('ollama.Client')
def test_ollama_invoke_streaming(mock_client_class, sample_messages, sample_system_prompt):
    """Test invoke with streaming."""
    mock_client = Mock()
    mock_stream = [
        {'message': {'content': 'Hello '}},
        {'message': {'content': 'world'}},
        {'message': {'content': '!'}}
    ]
    mock_client.chat.return_value = iter(mock_stream)
    mock_client_class.return_value = mock_client

    model = OllamaInference()
    response = model.invoke(sample_messages, sample_system_prompt, streaming=True)

    # Consume the generator
    result = ''.join(list(response))
    assert result == 'Hello world!'


@patch('ollama.Client')
def test_ollama_invoke_with_images(mock_client_class, sample_messages, sample_system_prompt, sample_image_file):
    """Test invoke with images."""
    mock_client = Mock()
    mock_client.chat.return_value = {'message': {'content': 'I see an image'}}
    mock_client_class.return_value = mock_client

    model = OllamaInference()
    response = model.invoke(
        sample_messages,
        sample_system_prompt,
        images=[sample_image_file]
    )

    assert response == 'I see an image'
    call_args = mock_client.chat.call_args
    # Check that images were added to the last message
    messages = call_args[1]['messages']
    assert 'images' in messages[-1]
    assert len(messages[-1]['images']) == 1


@patch('ollama.Client')
def test_ollama_invoke_with_schema(mock_client_class, sample_messages, sample_system_prompt):
    """Test invoke with schema."""
    from pydantic import BaseModel

    class TestSchema(BaseModel):
        name: str
        age: int

    mock_client = Mock()
    mock_client.chat.return_value = {'message': {'content': '{"name": "John", "age": 30}'}}
    mock_client_class.return_value = mock_client

    model = OllamaInference()
    response = model.invoke(sample_messages, sample_system_prompt, schema=TestSchema)

    assert '{"name": "John", "age": 30}' in response
    call_args = mock_client.chat.call_args
    assert 'format' in call_args[1]


@patch('ollama.Client')
def test_ollama_close(mock_client_class):
    """Test closing Ollama client."""
    mock_client = Mock()
    mock_client._client = Mock()
    mock_client_class.return_value = mock_client

    model = OllamaInference()
    model.close()

    mock_client._client.close.assert_called_once()