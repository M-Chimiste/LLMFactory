"""Tests for AnthropicInference."""
import pytest
from unittest.mock import Mock, patch
from LLMFactory.llm import AnthropicInference


@patch('anthropic.Anthropic')
def test_anthropic_init(mock_anthropic_class, mock_env_vars):
    """Test AnthropicInference initialization."""
    mock_anthropic_class.return_value = Mock()
    model = AnthropicInference(model_name="claude-3-opus")

    assert model.model_name == "claude-3-opus"
    assert model.provider == "anthropic"
    mock_anthropic_class.assert_called_once()


@patch('anthropic.Anthropic')
def test_anthropic_invoke_basic(mock_anthropic_class, mock_env_vars, sample_messages, sample_system_prompt):
    """Test basic invoke without streaming."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text="Claude response")]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_class.return_value = mock_client

    model = AnthropicInference(model_name="claude-3-opus")
    response = model.invoke(sample_messages, sample_system_prompt)

    assert response == "Claude response"
    mock_client.messages.create.assert_called_once()


@patch('anthropic.Anthropic')
def test_anthropic_invoke_streaming(mock_anthropic_class, mock_env_vars, sample_messages, sample_system_prompt):
    """Test invoke with streaming."""
    mock_client = Mock()

    # Create mock stream events
    mock_events = [
        Mock(type="content_block_delta", delta={"text": "Hello "}),
        Mock(type="content_block_delta", delta={"text": "from "}),
        Mock(type="content_block_delta", delta={"text": "Claude"}),
        Mock(type="other", delta={}),
    ]
    mock_client.messages.create.return_value = iter(mock_events)
    mock_anthropic_class.return_value = mock_client

    model = AnthropicInference(model_name="claude-3-opus")
    response = model.invoke(sample_messages, sample_system_prompt, streaming=True)

    result = ''.join(list(response))
    assert result == "Hello from Claude"


@patch('anthropic.Anthropic')
def test_anthropic_invoke_with_images(mock_anthropic_class, mock_env_vars, sample_messages, sample_system_prompt, sample_image_file):
    """Test invoke with images."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text="I see the image")]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_class.return_value = mock_client

    model = AnthropicInference(model_name="claude-3-opus")
    response = model.invoke(
        sample_messages,
        sample_system_prompt,
        images=[sample_image_file]
    )

    assert response == "I see the image"
    call_args = mock_client.messages.create.call_args
    messages = call_args[1]['messages']

    # Check that the last message has content array with image
    assert isinstance(messages[-1]['content'], list)
    assert any(item['type'] == 'image' for item in messages[-1]['content'])


@patch('anthropic.Anthropic')
def test_anthropic_custom_model_name(mock_anthropic_class, mock_env_vars, sample_messages, sample_system_prompt):
    """Test invoke with custom model name override."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text="Response")]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_class.return_value = mock_client

    model = AnthropicInference(model_name="claude-3-opus")
    response = model.invoke(sample_messages, sample_system_prompt, model_name="claude-3-sonnet")

    call_args = mock_client.messages.create.call_args
    assert call_args[1]['model'] == "claude-3-sonnet"