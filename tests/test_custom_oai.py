"""Tests for CustomOAIInference."""
import pytest
from unittest.mock import Mock, patch
from LLMFactory.llm import CustomOAIInference


@patch('openai.OpenAI')
def test_custom_oai_init(mock_openai_class, mock_env_vars):
    """Test CustomOAIInference initialization."""
    mock_openai_class.return_value = Mock()

    model = CustomOAIInference(model_name="custom-model")

    assert model.model_name == "custom-model"
    assert model.provider == "custom-oai"
    assert model.base_url == "http://localhost:8000"
    mock_openai_class.assert_called_once_with(
        base_url="http://localhost:8000",
        api_key="test-custom-key"
    )


@patch('openai.OpenAI')
def test_custom_oai_init_no_base_url(mock_openai_class, monkeypatch):
    """Test CustomOAIInference initialization fails without base URL."""
    monkeypatch.delenv("CUSTOM_OAI_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="CUSTOM_OAI_BASE_URL"):
        CustomOAIInference(model_name="custom-model")


@patch('openai.OpenAI')
def test_custom_oai_init_custom_params(mock_openai_class):
    """Test CustomOAIInference with custom base_url and api_key."""
    mock_openai_class.return_value = Mock()

    model = CustomOAIInference(
        model_name="custom-model",
        base_url="http://custom:9000",
        api_key="custom-key"
    )

    assert model.base_url == "http://custom:9000"
    assert model.api_key == "custom-key"
    mock_openai_class.assert_called_once_with(
        base_url="http://custom:9000",
        api_key="custom-key"
    )


@patch('openai.OpenAI')
def test_custom_oai_invoke_basic(mock_openai_class, mock_env_vars, sample_messages, sample_system_prompt):
    """Test basic invoke without streaming."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Custom response"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    model = CustomOAIInference(model_name="custom-model")
    response = model.invoke(sample_messages, sample_system_prompt)

    assert response == "Custom response"
    mock_client.chat.completions.create.assert_called_once()


@patch('openai.OpenAI')
def test_custom_oai_invoke_streaming(mock_openai_class, mock_env_vars, sample_messages, sample_system_prompt):
    """Test invoke with streaming."""
    mock_client = Mock()

    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Hello "))]),
        Mock(choices=[Mock(delta=Mock(content="custom "))]),
        Mock(choices=[Mock(delta=Mock(content="server"))]),
        Mock(choices=[Mock(delta=Mock(content=None))]),
    ]
    mock_client.chat.completions.create.return_value = iter(mock_chunks)
    mock_openai_class.return_value = mock_client

    model = CustomOAIInference(model_name="custom-model")
    response = model.invoke(sample_messages, sample_system_prompt, streaming=True)

    result = ''.join(list(response))
    assert result == "Hello custom server"


@patch('openai.OpenAI')
def test_custom_oai_invoke_with_images(mock_openai_class, mock_env_vars, sample_messages, sample_system_prompt, sample_image_file):
    """Test invoke with images."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="I see the image"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    model = CustomOAIInference(model_name="custom-vision-model")
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
def test_custom_oai_invoke_with_schema(mock_openai_class, mock_env_vars, sample_messages, sample_system_prompt):
    """Test invoke with schema."""
    from pydantic import BaseModel

    class TestSchema(BaseModel):
        name: str
        age: int

    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='{"name": "Bob", "age": 40}'))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    model = CustomOAIInference(model_name="custom-model")
    response = model.invoke(sample_messages, sample_system_prompt, schema=TestSchema)

    assert '{"name": "Bob", "age": 40}' in response
    call_args = mock_client.chat.completions.create.call_args
    assert 'response_format' in call_args[1]
    assert call_args[1]['response_format']['type'] == 'json_schema'


@patch('openai.OpenAI')
def test_custom_oai_no_api_key(mock_openai_class):
    """Test CustomOAIInference works without API key (for local servers)."""
    mock_openai_class.return_value = Mock()

    model = CustomOAIInference(
        model_name="custom-model",
        base_url="http://localhost:8000",
        api_key=None
    )

    assert model.api_key is None
    mock_openai_class.assert_called_once_with(
        base_url="http://localhost:8000",
        api_key=None
    )