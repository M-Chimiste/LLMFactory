"""Tests for OpenAIInference."""
import pytest
from unittest.mock import Mock, patch
from LLMFactory.llm import OpenAIInference


@patch('openai.OpenAI')
def test_openai_init(mock_openai_class, mock_env_vars):
    """Test OpenAIInference initialization."""
    mock_openai_class.return_value = Mock()
    model = OpenAIInference(model_name="gpt-4")

    assert model.model_name == "gpt-4"
    assert model.provider == "openai"
    mock_openai_class.assert_called_once()


@patch('openai.OpenAI')
def test_openai_invoke_basic(mock_openai_class, mock_env_vars, sample_messages, sample_system_prompt):
    """Test basic invoke without streaming."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="GPT response"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    model = OpenAIInference(model_name="gpt-4")
    response = model.invoke(sample_messages, sample_system_prompt)

    assert response == "GPT response"
    mock_client.chat.completions.create.assert_called_once()


@patch('openai.OpenAI')
def test_openai_invoke_streaming(mock_openai_class, mock_env_vars, sample_messages, sample_system_prompt):
    """Test invoke with streaming."""
    mock_client = Mock()

    # Create mock stream chunks
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Hello "))]),
        Mock(choices=[Mock(delta=Mock(content="from "))]),
        Mock(choices=[Mock(delta=Mock(content="GPT"))]),
        Mock(choices=[Mock(delta=Mock(content=None))]),
    ]
    mock_client.chat.completions.create.return_value = iter(mock_chunks)
    mock_openai_class.return_value = mock_client

    model = OpenAIInference(model_name="gpt-4")
    response = model.invoke(sample_messages, sample_system_prompt, streaming=True)

    result = ''.join(list(response))
    assert result == "Hello from GPT"


@patch('openai.OpenAI')
def test_openai_invoke_with_images(mock_openai_class, mock_env_vars, sample_messages, sample_system_prompt, sample_image_file):
    """Test invoke with images."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="I see the image"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    model = OpenAIInference(model_name="gpt-4-vision")
    response = model.invoke(
        sample_messages,
        sample_system_prompt,
        images=[sample_image_file]
    )

    assert response == "I see the image"
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args[1]['messages']

    # Check that the last message has content array with image_url
    assert isinstance(messages[-1]['content'], list)
    assert any(item['type'] == 'image_url' for item in messages[-1]['content'])


@patch('openai.OpenAI')
def test_openai_invoke_with_schema(mock_openai_class, mock_env_vars, sample_messages, sample_system_prompt):
    """Test invoke with schema."""
    from pydantic import BaseModel

    class TestSchema(BaseModel):
        name: str
        age: int

    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='{"name": "John", "age": 30}'))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    model = OpenAIInference(model_name="gpt-4")
    response = model.invoke(sample_messages, sample_system_prompt, schema=TestSchema)

    assert '{"name": "John", "age": 30}' in response
    call_args = mock_client.chat.completions.create.call_args
    assert 'response_format' in call_args[1]
    assert call_args[1]['response_format']['type'] == 'json_schema'


@patch('openai.OpenAI')
def test_openai_custom_model_name(mock_openai_class, mock_env_vars, sample_messages, sample_system_prompt):
    """Test invoke with custom model name override."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Response"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    model = OpenAIInference(model_name="gpt-4")
    response = model.invoke(sample_messages, sample_system_prompt, model_name="gpt-3.5-turbo")

    call_args = mock_client.chat.completions.create.call_args
    assert call_args[1]['model'] == "gpt-3.5-turbo"