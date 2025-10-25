"""Tests for AnthropicBedrockInference."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from LLMFactory.llm import AnthropicBedrockInference


@patch('boto3.Session')
@patch('anthropic.AnthropicBedrock')
def test_bedrock_init_with_profile(mock_bedrock_class, mock_session_class):
    """Test AnthropicBedrockInference initialization with AWS profile."""
    # Mock boto3 session and credentials
    mock_creds = Mock()
    mock_creds.access_key = 'test-access-key'
    mock_creds.secret_key = 'test-secret-key'
    mock_creds.token = 'test-session-token'

    mock_session = Mock()
    mock_session.get_credentials.return_value = mock_creds
    mock_session_class.return_value = mock_session

    mock_bedrock_class.return_value = Mock()

    model = AnthropicBedrockInference(
        model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        aws_profile="test-profile",
        region_name="us-west-2"
    )

    assert model.model_name == "anthropic.claude-3-sonnet-20240229-v1:0"
    assert model.provider == "anthropic-bedrock"
    assert model.aws_profile == "test-profile"
    assert model.region_name == "us-west-2"

    # Verify AnthropicBedrock was called with correct credentials
    mock_bedrock_class.assert_called_once_with(
        aws_access_key='test-access-key',
        aws_secret_key='test-secret-key',
        aws_session_token='test-session-token',
        aws_region='us-west-2'
    )


@patch('boto3.Session')
@patch('anthropic.AnthropicBedrock')
def test_bedrock_init_with_env_vars(mock_bedrock_class, mock_session_class, monkeypatch):
    """Test AnthropicBedrockInference initialization with environment variables."""
    # Set environment variables
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret-key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "env-session-token")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")

    mock_bedrock_class.return_value = Mock()

    model = AnthropicBedrockInference()

    assert model.region_name == "eu-west-1"

    # Verify AnthropicBedrock was called with env var credentials
    mock_bedrock_class.assert_called_once_with(
        aws_access_key='env-access-key',
        aws_secret_key='env-secret-key',
        aws_session_token='env-session-token',
        aws_region='eu-west-1'
    )


@patch('boto3.Session')
@patch('anthropic.AnthropicBedrock')
def test_bedrock_init_with_default_session(mock_bedrock_class, mock_session_class, monkeypatch):
    """Test AnthropicBedrockInference initialization with boto3 default session."""
    # Remove env vars
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    # Mock boto3 default session credentials
    mock_creds = Mock()
    mock_creds.access_key = 'default-access-key'
    mock_creds.secret_key = 'default-secret-key'
    mock_creds.token = 'default-session-token'

    mock_session = Mock()
    mock_session.get_credentials.return_value = mock_creds
    mock_session_class.return_value = mock_session

    mock_bedrock_class.return_value = Mock()

    model = AnthropicBedrockInference()

    # Verify AnthropicBedrock was called with default session credentials
    mock_bedrock_class.assert_called_once_with(
        aws_access_key='default-access-key',
        aws_secret_key='default-secret-key',
        aws_session_token='default-session-token',
        aws_region='us-east-1'
    )


@patch('boto3.Session')
def test_bedrock_init_no_credentials(mock_session_class, monkeypatch):
    """Test AnthropicBedrockInference initialization fails without credentials."""
    # Remove all env vars
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    # Mock boto3 session to return no credentials
    mock_session = Mock()
    mock_session.get_credentials.return_value = None
    mock_session_class.return_value = mock_session

    with pytest.raises(ValueError, match="No valid AWS credentials found"):
        AnthropicBedrockInference()


@patch('boto3.Session')
@patch('anthropic.AnthropicBedrock')
def test_bedrock_invoke_basic(mock_bedrock_class, mock_session_class, monkeypatch, sample_messages, sample_system_prompt):
    """Test basic invoke without streaming."""
    # Setup credentials
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")

    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text="Bedrock Claude response")]
    mock_client.messages.create.return_value = mock_response
    mock_bedrock_class.return_value = mock_client

    model = AnthropicBedrockInference(
        model_name="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    response = model.invoke(sample_messages, sample_system_prompt)

    assert response == "Bedrock Claude response"
    mock_client.messages.create.assert_called_once()
    call_args = mock_client.messages.create.call_args
    assert call_args[1]['model'] == "anthropic.claude-3-sonnet-20240229-v1:0"


@patch('boto3.Session')
@patch('anthropic.AnthropicBedrock')
def test_bedrock_invoke_streaming(mock_bedrock_class, mock_session_class, monkeypatch, sample_messages, sample_system_prompt):
    """Test invoke with streaming."""
    # Setup credentials
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")

    mock_client = Mock()

    # Create mock stream events
    mock_events = [
        Mock(type="content_block_delta", delta={"text": "Hello "}),
        Mock(type="content_block_delta", delta={"text": "from "}),
        Mock(type="content_block_delta", delta={"text": "Bedrock"}),
        Mock(type="other", delta={}),
    ]
    mock_client.messages.create.return_value = iter(mock_events)
    mock_bedrock_class.return_value = mock_client

    model = AnthropicBedrockInference()
    response = model.invoke(sample_messages, sample_system_prompt, streaming=True)

    result = ''.join(list(response))
    assert result == "Hello from Bedrock"


@patch('boto3.Session')
@patch('anthropic.AnthropicBedrock')
def test_bedrock_invoke_with_images(mock_bedrock_class, mock_session_class, monkeypatch, sample_messages, sample_system_prompt, sample_image_file):
    """Test invoke with images."""
    # Setup credentials
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")

    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text="I see the image")]
    mock_client.messages.create.return_value = mock_response
    mock_bedrock_class.return_value = mock_client

    model = AnthropicBedrockInference()
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


@patch('boto3.Session')
@patch('anthropic.AnthropicBedrock')
def test_bedrock_custom_model_name(mock_bedrock_class, mock_session_class, monkeypatch, sample_messages, sample_system_prompt):
    """Test invoke with custom model name override."""
    # Setup credentials
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")

    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text="Response")]
    mock_client.messages.create.return_value = mock_response
    mock_bedrock_class.return_value = mock_client

    model = AnthropicBedrockInference(
        model_name="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    response = model.invoke(
        sample_messages,
        sample_system_prompt,
        model_name="anthropic.claude-3-haiku-20240307-v1:0"
    )

    call_args = mock_client.messages.create.call_args
    assert call_args[1]['model'] == "anthropic.claude-3-haiku-20240307-v1:0"


@patch('boto3.Session')
@patch('anthropic.AnthropicBedrock')
def test_bedrock_default_region(mock_bedrock_class, mock_session_class, monkeypatch):
    """Test that default region is us-east-1."""
    # Setup credentials without region
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.delenv("AWS_REGION", raising=False)

    mock_bedrock_class.return_value = Mock()

    model = AnthropicBedrockInference()

    assert model.region_name == "us-east-1"

    # Verify AnthropicBedrock was called with us-east-1
    call_args = mock_bedrock_class.call_args
    assert call_args[1]['aws_region'] == 'us-east-1'


@patch('boto3.Session')
def test_bedrock_profile_not_found(mock_session_class, monkeypatch):
    """Test handling of ProfileNotFound exception."""
    from botocore.exceptions import ProfileNotFound

    # Remove env vars
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    # Mock boto3 to raise ProfileNotFound for the specified profile
    def session_side_effect(profile_name=None, region_name=None):
        if profile_name == "nonexistent-profile":
            raise ProfileNotFound(profile="nonexistent-profile")
        mock_session = Mock()
        mock_session.get_credentials.return_value = None
        return mock_session

    mock_session_class.side_effect = session_side_effect

    with pytest.raises(ValueError, match="No valid AWS credentials found"):
        AnthropicBedrockInference(aws_profile="nonexistent-profile")