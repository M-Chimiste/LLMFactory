"""Tests for LMStudioInference."""
import pytest
import sys
from unittest.mock import Mock, patch
from LLMFactory.llm import LMStudioInference


@pytest.fixture(autouse=True)
def setup_lmstudio_mock():
    """Setup LMStudio mock for all tests."""
    mock_lms_module = Mock()
    mock_lms_module.Client.is_valid_api_host.return_value = True
    sys.modules['lmstudio'] = mock_lms_module
    yield mock_lms_module
    # Cleanup after each test
    if 'lmstudio' in sys.modules:
        del sys.modules['lmstudio']


def test_lmstudio_init_default(setup_lmstudio_mock):
    """Test LMStudioInference initialization with defaults."""
    model = LMStudioInference(model_name="qwen2.5-7b-instruct")

    assert model.model_name == "qwen2.5-7b-instruct"
    assert model.max_new_tokens == 4096
    assert model.temperature == 0.1
    assert model.host == "localhost:1234"
    assert model.provider == "lmstudio"
    assert model.context_length is None
    assert model.gpu_offload is None


def test_lmstudio_init_custom(setup_lmstudio_mock):
    """Test LMStudioInference initialization with custom parameters."""
    model = LMStudioInference(
        model_name="llama-3.1-8b",
        max_new_tokens=2048,
        temperature=0.7,
        host="athena.local:1234",
        context_length=32768,
        gpu_offload="max"
    )

    assert model.model_name == "llama-3.1-8b"
    assert model.max_new_tokens == 2048
    assert model.temperature == 0.7
    assert model.host == "athena.local:1234"
    assert model.context_length == 32768
    assert model.gpu_offload == "max"


def test_lmstudio_init_with_env_var(setup_lmstudio_mock, monkeypatch):
    """Test LMStudioInference initialization with environment variable."""
    monkeypatch.setenv("LMSTUDIO_HOST", "remote.server:5678")

    model = LMStudioInference(model_name="test-model")

    assert model.host == "remote.server:5678"


def test_lmstudio_init_connection_error():
    """Test LMStudioInference initialization with connection error."""
    mock_lms_module = Mock()
    mock_lms_module.Client.is_valid_api_host.return_value = False
    sys.modules['lmstudio'] = mock_lms_module

    with pytest.raises(ConnectionError, match="Cannot connect to LM Studio"):
        LMStudioInference(model_name="test-model")

    del sys.modules['lmstudio']


def test_lmstudio_init_import_error():
    """Test LMStudioInference initialization with missing lmstudio package."""
    # Remove lmstudio from sys.modules if it exists
    if 'lmstudio' in sys.modules:
        del sys.modules['lmstudio']

    with patch.dict('sys.modules', {'lmstudio': None}):
        with pytest.raises(ImportError, match="lmstudio-python is not installed"):
            LMStudioInference(model_name="test-model")


def test_lmstudio_invoke_basic(setup_lmstudio_mock, sample_messages, sample_system_prompt):
    """Test basic invoke without streaming."""
    # Setup mocks
    mock_model = Mock()
    mock_response = Mock()
    mock_response.content = "Test response"
    mock_model.respond.return_value = mock_response
    setup_lmstudio_mock.llm.return_value = mock_model
    mock_chat = Mock()
    setup_lmstudio_mock.Chat.return_value = mock_chat

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(sample_messages, sample_system_prompt)

    assert response == "Test response"
    mock_chat.add_system_message.assert_called_with(sample_system_prompt)
    assert mock_chat.add_user_message.call_count == 2
    assert mock_chat.add_assistant_message.call_count == 1
    mock_model.respond.assert_called_once()


def test_lmstudio_invoke_streaming(setup_lmstudio_mock, sample_messages, sample_system_prompt):
    """Test invoke with streaming."""
    # Setup mocks
    mock_model = Mock()

    # Create streaming chunks
    mock_chunks = [
        Mock(content="Hello "),
        Mock(content="world"),
        Mock(content="!")
    ]
    mock_model.respond_stream.return_value = iter(mock_chunks)
    setup_lmstudio_mock.llm.return_value = mock_model
    mock_chat = Mock()
    setup_lmstudio_mock.Chat.return_value = mock_chat

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(sample_messages, sample_system_prompt, streaming=True)

    # Consume the generator
    result = ''.join(list(response))
    assert result == "Hello world!"

    mock_model.respond_stream.assert_called_once()


def test_lmstudio_invoke_with_images(setup_lmstudio_mock, sample_messages, sample_system_prompt, sample_image_file):
    """Test invoke with images."""
    # Setup mocks
    mock_model = Mock()
    mock_response = Mock()
    mock_response.content = "I see an image"
    mock_model.respond.return_value = mock_response
    setup_lmstudio_mock.llm.return_value = mock_model
    mock_chat = Mock()
    setup_lmstudio_mock.Chat.return_value = mock_chat
    mock_image_handle = Mock()
    setup_lmstudio_mock.prepare_image.return_value = mock_image_handle

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(
        sample_messages,
        sample_system_prompt,
        images=[sample_image_file]
    )

    assert response == "I see an image"
    setup_lmstudio_mock.prepare_image.assert_called_once_with(sample_image_file)
    # Check that images were passed to the last user message
    mock_chat.add_user_message.assert_called()
    last_call = mock_chat.add_user_message.call_args_list[-1]
    assert 'images' in last_call[1]
    assert mock_image_handle in last_call[1]['images']


def test_lmstudio_invoke_with_images_bytes(setup_lmstudio_mock, sample_messages, sample_system_prompt, sample_image_bytes):
    """Test invoke with images as bytes."""
    # Setup mocks
    mock_model = Mock()
    mock_response = Mock()
    mock_response.content = "I see an image"
    mock_model.respond.return_value = mock_response
    setup_lmstudio_mock.llm.return_value = mock_model
    mock_chat = Mock()
    setup_lmstudio_mock.Chat.return_value = mock_chat
    mock_image_handle = Mock()
    setup_lmstudio_mock.prepare_image.return_value = mock_image_handle

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(
        sample_messages,
        sample_system_prompt,
        images=[sample_image_bytes]
    )

    assert response == "I see an image"
    setup_lmstudio_mock.prepare_image.assert_called_once_with(sample_image_bytes)


def test_lmstudio_invoke_with_schema(setup_lmstudio_mock, sample_messages, sample_system_prompt):
    """Test invoke with schema."""
    from pydantic import BaseModel

    class TestSchema(BaseModel):
        name: str
        age: int

    # Setup mocks
    mock_model = Mock()
    mock_response = Mock()
    mock_response.content = '{"name": "John", "age": 30}'
    mock_model.respond.return_value = mock_response
    setup_lmstudio_mock.llm.return_value = mock_model
    mock_chat = Mock()
    setup_lmstudio_mock.Chat.return_value = mock_chat

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(sample_messages, sample_system_prompt, schema=TestSchema)

    assert '{"name": "John", "age": 30}' in response
    call_args = mock_model.respond.call_args
    assert 'response_format' in call_args[1]
    assert call_args[1]['response_format'] == TestSchema


def test_lmstudio_invoke_with_custom_params(setup_lmstudio_mock, sample_messages, sample_system_prompt):
    """Test invoke with custom parameters."""
    # Setup mocks
    mock_model = Mock()
    mock_response = Mock()
    mock_response.content = "Test response"
    mock_model.respond.return_value = mock_response
    setup_lmstudio_mock.llm.return_value = mock_model
    mock_chat = Mock()
    setup_lmstudio_mock.Chat.return_value = mock_chat

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(
        sample_messages,
        sample_system_prompt,
        max_tokens=1024,
        temperature=0.9,
        top_p=0.95,
        top_k=50
    )

    assert response == "Test response"
    call_args = mock_model.respond.call_args[1]
    assert call_args['max_tokens'] == 1024
    assert call_args['temperature'] == 0.9
    assert call_args['top_p'] == 0.95
    assert call_args['top_k'] == 50


def test_lmstudio_invoke_response_dict_format(setup_lmstudio_mock, sample_messages, sample_system_prompt):
    """Test invoke handling dict response format."""
    # Setup mocks
    mock_model = Mock()
    mock_model.respond.return_value = {'content': 'Dict response'}
    setup_lmstudio_mock.llm.return_value = mock_model
    mock_chat = Mock()
    setup_lmstudio_mock.Chat.return_value = mock_chat

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(sample_messages, sample_system_prompt)

    assert response == "Dict response"


def test_lmstudio_invoke_error_handling(setup_lmstudio_mock, sample_messages, sample_system_prompt):
    """Test invoke error handling."""
    # Setup mocks
    mock_model = Mock()
    mock_model.respond.side_effect = Exception("Model error")
    setup_lmstudio_mock.llm.return_value = mock_model
    mock_chat = Mock()
    setup_lmstudio_mock.Chat.return_value = mock_chat

    model = LMStudioInference(model_name="test-model")
    with pytest.raises(RuntimeError, match="Error during LM Studio inference"):
        model.invoke(sample_messages, sample_system_prompt)


def test_lmstudio_get_or_load_model_with_config(setup_lmstudio_mock):
    """Test model loading with configuration."""
    # Setup mocks
    mock_model = Mock()
    setup_lmstudio_mock.llm.return_value = mock_model

    model = LMStudioInference(
        model_name="test-model",
        context_length=32768,
        gpu_offload="max"
    )
    loaded_model = model._get_or_load_model()

    assert loaded_model == mock_model
    call_args = setup_lmstudio_mock.llm.call_args
    assert call_args[0][0] == "test-model"
    assert 'config' in call_args[1]
    assert call_args[1]['config']['context_length'] == 32768
    assert call_args[1]['config']['gpu_offload'] == "max"


def test_lmstudio_get_or_load_model_with_gpu_ratio(setup_lmstudio_mock):
    """Test model loading with GPU ratio."""
    # Setup mocks
    mock_model = Mock()
    setup_lmstudio_mock.llm.return_value = mock_model

    model = LMStudioInference(
        model_name="test-model",
        gpu_offload=0.75
    )
    loaded_model = model._get_or_load_model()

    assert loaded_model == mock_model
    call_args = setup_lmstudio_mock.llm.call_args
    assert 'config' in call_args[1]
    assert call_args[1]['config']['gpu'] == {'ratio': 0.75}


def test_lmstudio_get_or_load_model_caching(setup_lmstudio_mock, sample_messages, sample_system_prompt):
    """Test that model is cached and not loaded multiple times."""
    # Setup mocks
    mock_model = Mock()
    mock_response = Mock()
    mock_response.content = "Test response"
    mock_model.respond.return_value = mock_response
    setup_lmstudio_mock.llm.return_value = mock_model
    mock_chat = Mock()
    setup_lmstudio_mock.Chat.return_value = mock_chat

    model = LMStudioInference(model_name="test-model")

    # First invoke
    model.invoke(sample_messages, sample_system_prompt)
    # Second invoke
    model.invoke(sample_messages, sample_system_prompt)

    # Model should only be loaded once
    assert setup_lmstudio_mock.llm.call_count == 1


def test_lmstudio_unload_model(setup_lmstudio_mock):
    """Test model unloading."""
    # Setup mocks
    mock_model = Mock()
    setup_lmstudio_mock.llm.return_value = mock_model

    model = LMStudioInference(model_name="test-model")
    model._get_or_load_model()

    assert model._model_instance is not None
    model.unload_model()
    assert model._model_instance is None
    mock_model.unload.assert_called_once()


def test_lmstudio_unload_model_error_handling(setup_lmstudio_mock):
    """Test model unloading with error."""
    # Setup mocks
    mock_model = Mock()
    mock_model.unload.side_effect = Exception("Unload error")
    setup_lmstudio_mock.llm.return_value = mock_model

    model = LMStudioInference(model_name="test-model")
    model._get_or_load_model()

    # Should not raise exception
    model.unload_model()
    assert model._model_instance is None


def test_lmstudio_close(setup_lmstudio_mock):
    """Test closing LMStudio client."""
    # Setup mocks
    mock_model = Mock()
    setup_lmstudio_mock.llm.return_value = mock_model

    model = LMStudioInference(model_name="test-model")
    model._get_or_load_model()

    model.close()
    mock_model.unload.assert_called_once()


def test_lmstudio_streaming_dict_chunks(setup_lmstudio_mock, sample_messages, sample_system_prompt):
    """Test streaming with dict-format chunks."""
    # Setup mocks
    mock_model = Mock()

    # Create streaming chunks as dicts
    mock_chunks = [
        {'content': 'Test '},
        {'content': 'streaming'},
        {'content': ''}  # Empty chunk should be filtered
    ]
    mock_model.respond_stream.return_value = iter(mock_chunks)
    setup_lmstudio_mock.llm.return_value = mock_model
    mock_chat = Mock()
    setup_lmstudio_mock.Chat.return_value = mock_chat

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(sample_messages, sample_system_prompt, streaming=True)

    result = ''.join(list(response))
    assert result == "Test streaming"


def test_lmstudio_streaming_string_chunks(setup_lmstudio_mock, sample_messages, sample_system_prompt):
    """Test streaming with string-format chunks."""
    # Setup mocks
    mock_model = Mock()

    # Create streaming chunks as strings
    mock_chunks = ['Hello', ' ', 'LMStudio']
    mock_model.respond_stream.return_value = iter(mock_chunks)
    setup_lmstudio_mock.llm.return_value = mock_model
    mock_chat = Mock()
    setup_lmstudio_mock.Chat.return_value = mock_chat

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(sample_messages, sample_system_prompt, streaming=True)

    result = ''.join(list(response))
    assert result == "Hello LMStudio"


def test_lmstudio_invalid_image_type(setup_lmstudio_mock, sample_messages, sample_system_prompt):
    """Test invoke with invalid image type."""
    # Setup mocks
    mock_model = Mock()
    setup_lmstudio_mock.llm.return_value = mock_model
    mock_chat = Mock()
    setup_lmstudio_mock.Chat.return_value = mock_chat

    model = LMStudioInference(model_name="test-model")
    with pytest.raises(ValueError, match="Unsupported image type"):
        model.invoke(
            sample_messages,
            sample_system_prompt,
            images=[123]  # Invalid type
        )
