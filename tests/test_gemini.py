"""Tests for GeminiInference."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from LLMFactory.llm import GeminiInference


@patch('google.generativeai.configure')
@patch('google.generativeai.GenerativeModel')
def test_gemini_init(mock_model_class, mock_configure, mock_env_vars):
    """Test GeminiInference initialization."""
    mock_model_class.return_value = Mock()

    model = GeminiInference(model_name="gemini-pro")

    assert model.model_name == "gemini-pro"
    assert model.provider == "gemini"
    assert len(model.safety) == 4
    mock_configure.assert_called_once()


@patch('google.generativeai.configure')
@patch('google.generativeai.GenerativeModel')
@patch('google.generativeai.types.GenerationConfig')
def test_gemini_invoke_basic(mock_config, mock_model_class, mock_configure, mock_env_vars, sample_messages, sample_system_prompt):
    """Test basic invoke without streaming."""
    mock_model_instance = Mock()
    mock_response = Mock(text="Gemini response")
    mock_model_instance.generate_content.return_value = mock_response
    mock_model_class.return_value = mock_model_instance

    # Need to patch the client attribute
    with patch('LLMFactory.llm.GeminiInference._load_model') as mock_load:
        mock_client = MagicMock()
        mock_client.GenerativeModel.return_value = mock_model_instance
        mock_client.types.GenerationConfig = mock_config
        mock_load.return_value = mock_client

        model = GeminiInference(model_name="gemini-pro")
        response = model.invoke(sample_messages, sample_system_prompt)

        assert response == "Gemini response"


@patch('google.generativeai.configure')
@patch('google.generativeai.GenerativeModel')
@patch('google.generativeai.types.GenerationConfig')
def test_gemini_invoke_streaming(mock_config, mock_model_class, mock_configure, mock_env_vars, sample_messages, sample_system_prompt):
    """Test invoke with streaming."""
    mock_model_instance = Mock()

    # Create mock stream chunks
    mock_chunks = [
        Mock(text="Hello "),
        Mock(text="from "),
        Mock(text="Gemini"),
    ]
    mock_model_instance.generate_content.return_value = iter(mock_chunks)
    mock_model_class.return_value = mock_model_instance

    with patch('LLMFactory.llm.GeminiInference._load_model') as mock_load:
        mock_client = MagicMock()
        mock_client.GenerativeModel.return_value = mock_model_instance
        mock_client.types.GenerationConfig = mock_config
        mock_load.return_value = mock_client

        model = GeminiInference(model_name="gemini-pro")
        response = model.invoke(sample_messages, sample_system_prompt, streaming=True)

        result = ''.join(list(response))
        assert result == "Hello from Gemini"


@patch('google.generativeai.configure')
@patch('PIL.Image.open')
def test_gemini_invoke_with_images(mock_pil_open, mock_configure, mock_env_vars, sample_messages, sample_system_prompt, sample_image_file):
    """Test invoke with images."""
    from PIL import Image

    mock_image = Mock(spec=Image.Image)
    mock_pil_open.return_value = mock_image

    with patch('LLMFactory.llm.GeminiInference._load_model') as mock_load:
        mock_client = MagicMock()
        mock_model_instance = Mock()
        mock_response = Mock(text="I see the image")
        mock_model_instance.generate_content.return_value = mock_response
        mock_client.GenerativeModel.return_value = mock_model_instance
        mock_client.types.GenerationConfig = MagicMock()
        mock_load.return_value = mock_client

        model = GeminiInference(model_name="gemini-pro-vision")
        response = model.invoke(
            sample_messages,
            sample_system_prompt,
            images=[sample_image_file]
        )

        assert response == "I see the image"
        mock_pil_open.assert_called_once_with(sample_image_file)


@patch('google.generativeai.configure')
def test_gemini_role_mapping(mock_configure, mock_env_vars, sample_system_prompt):
    """Test that assistant role is mapped to model role."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"}
    ]

    with patch('LLMFactory.llm.GeminiInference._load_model') as mock_load:
        mock_client = MagicMock()
        mock_model_instance = Mock()
        mock_response = Mock(text="Response")
        mock_model_instance.generate_content.return_value = mock_response
        mock_client.GenerativeModel.return_value = mock_model_instance
        mock_client.types.GenerationConfig = MagicMock()
        mock_load.return_value = mock_client

        model = GeminiInference()
        model.invoke(messages, sample_system_prompt)

        # Check the call to generate_content
        call_args = mock_model_instance.generate_content.call_args
        gemini_messages = call_args[0][0]

        # Second message should have role "model" (mapped from "assistant")
        assert gemini_messages[2]["role"] == "model"