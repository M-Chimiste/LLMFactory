"""Pytest configuration and shared fixtures."""
import pytest
import os
import tempfile
from unittest.mock import Mock, MagicMock
from PIL import Image
import io


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for API keys."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")
    monkeypatch.setenv("LMSTUDIO_HOST", "localhost:1234")
    monkeypatch.setenv("CUSTOM_OAI_BASE_URL", "http://localhost:8000")
    monkeypatch.setenv("CUSTOM_OAI_API_KEY", "test-custom-key")


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What's the weather like?"}
    ]


@pytest.fixture
def sample_system_prompt():
    """Sample system prompt for testing."""
    return "You are a helpful assistant."


@pytest.fixture
def sample_image_bytes():
    """Create a sample image in bytes."""
    img = Image.new('RGB', (100, 100), color='red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf.read()


@pytest.fixture
def sample_image_file(tmp_path, sample_image_bytes):
    """Create a temporary image file."""
    image_path = tmp_path / "test_image.jpg"
    with open(image_path, 'wb') as f:
        f.write(sample_image_bytes)
    return str(image_path)


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client."""
    mock = Mock()
    mock.messages.create.return_value = Mock(
        content=[Mock(text="Test response")]
    )
    return mock


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock = Mock()
    mock.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Test response"))]
    )
    return mock


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client."""
    mock = Mock()
    mock.chat.return_value = {
        'message': {'content': 'Test response'}
    }
    mock.embeddings.return_value = {
        'embedding': [0.1] * 384
    }
    return mock


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client."""
    mock = Mock()
    mock_model = Mock()
    mock_model.generate_content.return_value = Mock(text="Test response")
    mock.GenerativeModel.return_value = mock_model
    mock.types.GenerationConfig = MagicMock()
    return mock


@pytest.fixture
def mock_llama_client():
    """Mock Llama cpp client."""
    mock = Mock()
    mock.create_chat_completion.return_value = {
        'choices': [{'message': {'content': 'Test response'}}]
    }
    return mock


@pytest.fixture
def mock_lmstudio_client():
    """Mock LM Studio client."""
    mock = Mock()
    mock_model = Mock()
    mock_response = Mock()
    mock_response.content = "Test response"
    mock_model.respond.return_value = mock_response
    mock.llm.return_value = mock_model
    mock.Chat.return_value = Mock()
    mock.Client.is_valid_api_host.return_value = True
    return mock