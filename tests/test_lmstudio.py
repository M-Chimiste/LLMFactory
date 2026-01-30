"""Tests for LMStudioInference."""
import pytest
import sys
import threading
from unittest.mock import Mock, patch
from LLMFactory.llm import LMStudioInference, LLMModelFactory
from LLMFactory.providers import lmstudio as lmstudio_module
from LLMFactory import llm as llm_module


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
    # Check that Chat was initialized with the system prompt
    setup_lmstudio_mock.Chat.assert_called_with(sample_system_prompt)
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
    # Add parsed attribute for structured output
    mock_parsed = TestSchema(name="John", age=30)
    mock_response.parsed = mock_parsed
    mock_model.respond.return_value = mock_response
    setup_lmstudio_mock.llm.return_value = mock_model
    mock_chat = Mock()
    setup_lmstudio_mock.Chat.return_value = mock_chat

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(sample_messages, sample_system_prompt, schema=TestSchema)

    # When schema is provided, should return the parsed object
    assert response == mock_parsed
    assert response.name == "John"
    assert response.age == 30
    
    call_args = mock_model.respond.call_args
    # response_format is inside the config dict
    assert 'config' in call_args[1]
    assert 'response_format' in call_args[1]['config']
    assert call_args[1]['config']['response_format'] == TestSchema


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
    # LM Studio SDK uses a 'config' dict with camelCase parameters
    assert 'config' in call_args
    config = call_args['config']
    assert config['temperature'] == 0.9
    assert config['maxTokens'] == 1024
    assert config['topP'] == 0.95
    assert config['topK'] == 50


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
    # LM Studio SDK uses camelCase parameter names
    assert call_args[1]['config']['contextLength'] == 32768
    assert call_args[1]['config']['gpuOffload'] == "max"


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


# =============================================================================
# Singleton Pattern Handling Tests
# =============================================================================

@pytest.fixture(autouse=True)
def reset_lmstudio_state():
    """Reset all LMStudio singleton state before and after each test."""
    # Reset state before test
    lmstudio_module._lmstudio_configured_host = None
    llm_module._lmstudio_cache.clear()
    
    yield
    
    # Reset state after test to ensure clean state for next test
    lmstudio_module._lmstudio_configured_host = None
    llm_module._lmstudio_cache.clear()


def test_lmstudio_multiple_instances_same_host(setup_lmstudio_mock, reset_lmstudio_state):
    """Test that multiple instances with same host work correctly (no crash)."""
    # First instance
    client1 = LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='model-a',
        host='localhost:1234'
    )
    assert client1 is not None
    
    # Second instance - different model, same host (should NOT crash!)
    client2 = LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='model-b',
        host='localhost:1234'
    )
    assert client2 is not None
    assert client2 is not client1  # Different models = different instances
    
    # Third instance - same as first (returns cached)
    client3 = LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='model-a',
        host='localhost:1234'
    )
    assert client3 is client1  # Should be cached


def test_lmstudio_api_identical_to_other_providers(setup_lmstudio_mock, reset_lmstudio_state):
    """Test that LMStudio API is identical to other providers - same create_model() call."""
    # This is the key test - same API, no special handling needed
    lmstudio_client = LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='test-model'
    )
    
    # Creating multiple instances should NOT require any special handling
    lmstudio_client2 = LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='test-model-2'
    )
    
    # Both should work without any singleton errors
    assert lmstudio_client is not None
    assert lmstudio_client2 is not None


def test_lmstudio_host_change_warning(setup_lmstudio_mock, reset_lmstudio_state):
    """Test that changing hosts produces a warning but doesn't crash."""
    # First instance
    client1 = LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='test-model',
        host='localhost:1234'
    )
    
    # Attempt with different host - should warn but not crash
    with pytest.warns(UserWarning, match="Cannot change host"):
        client2 = LLMModelFactory.create_model(
            model_type='lmstudio',
            model_name='other-model',
            host='different-host:1234'
        )
    
    # Should still work, using original host
    assert client2 is not None
    assert client2.host == 'localhost:1234'


def test_lmstudio_cached_instance_returned(setup_lmstudio_mock, reset_lmstudio_state):
    """Test that requesting same model returns cached instance."""
    client1 = LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='test-model',
        host='localhost:1234'
    )
    
    client2 = LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='test-model',
        host='localhost:1234'
    )
    
    # Should be the exact same instance
    assert client1 is client2


def test_lmstudio_clear_cache(setup_lmstudio_mock, reset_lmstudio_state):
    """Test cache clearing functionality."""
    # Create and cache an instance
    instance1 = LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='test-model',
        host='localhost:1234'
    )
    
    # Clear cache
    LLMModelFactory.clear_lmstudio_cache()
    
    # Next call should create a new instance
    instance2 = LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='test-model',
        host='localhost:1234'
    )
    
    # Should be different instances (though SDK host config remains)
    assert instance2 is not instance1


def test_lmstudio_get_configured_host(setup_lmstudio_mock, reset_lmstudio_state):
    """Test getting the configured host via module function."""
    # Initially None
    assert lmstudio_module._get_configured_host() is None
    
    # After creating an instance
    LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='test-model',
        host='localhost:1234'
    )
    
    assert lmstudio_module._get_configured_host() == 'localhost:1234'


def test_lmstudio_reset_configured_host(setup_lmstudio_mock, reset_lmstudio_state):
    """Test resetting the configured host (for testing purposes)."""
    # Create an instance to set the host
    LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='test-model',
        host='localhost:1234'
    )
    
    assert lmstudio_module._get_configured_host() == 'localhost:1234'
    
    # Reset the host tracking
    lmstudio_module._reset_configured_host()
    
    assert lmstudio_module._get_configured_host() is None


def test_lmstudio_thread_safety(setup_lmstudio_mock, reset_lmstudio_state):
    """Test thread safety of the caching mechanism."""
    results = []
    errors = []
    
    def create_instance(model_name):
        try:
            instance = LLMModelFactory.create_model(
                model_type='lmstudio',
                model_name=model_name,
                host='localhost:1234'
            )
            results.append((model_name, instance))
        except Exception as e:
            errors.append(e)
    
    # Create multiple threads
    threads = []
    for i in range(10):
        t = threading.Thread(target=create_instance, args=(f'model-{i % 3}',))
        threads.append(t)
    
    # Start all threads
    for t in threads:
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    # Should have no errors
    assert len(errors) == 0
    
    # Instances with same model name should be identical
    model_instances = {}
    for model_name, instance in results:
        if model_name not in model_instances:
            model_instances[model_name] = instance
        else:
            assert model_instances[model_name] is instance


def test_lmstudio_direct_instantiation_singleton_handling(setup_lmstudio_mock, reset_lmstudio_state):
    """Test that direct instantiation also handles the singleton pattern."""
    # Direct instantiation (not through factory)
    model1 = LMStudioInference(model_name="test-model", host="localhost:1234")
    
    # Second direct instantiation - should not crash
    model2 = LMStudioInference(model_name="test-model-2", host="localhost:1234")
    
    assert model1 is not None
    assert model2 is not None
    # Direct instantiation doesn't use the factory cache, so these are different instances
    assert model1 is not model2


def test_lmstudio_configure_default_client_called_once(setup_lmstudio_mock, reset_lmstudio_state):
    """Test that configure_default_client is only called once."""
    # Create first instance
    LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='model-a',
        host='localhost:1234'
    )
    
    # Create second instance with same host
    LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='model-b',
        host='localhost:1234'
    )
    
    # configure_default_client should only be called once
    assert setup_lmstudio_mock.configure_default_client.call_count == 1


# =============================================================================
# Thinking Mode Tests
# =============================================================================

def test_lmstudio_thinking_returns_thinking_response(setup_lmstudio_mock):
    """Test that use_thinking=True with return_thinking=True returns ThinkingResponse."""
    from LLMFactory.llm import ThinkingResponse

    mock_response = Mock()
    mock_response.json.return_value = {
        'output': [
            {'type': 'reasoning', 'content': [{'text': 'Step by step reasoning...'}]},
            {'type': 'message', 'content': [{'text': 'The answer is 42.'}]}
        ]
    }
    mock_response.raise_for_status = Mock()

    with patch('LLMFactory.providers.lmstudio.requests.post', return_value=mock_response) as mock_post:
        model = LMStudioInference(model_name="test-model")
        response = model.invoke(
            [{"role": "user", "content": "Question"}],
            "System prompt",
            use_thinking=True,
            return_thinking=True
        )

        assert isinstance(response, ThinkingResponse)
        assert response.content == 'The answer is 42.'
        assert response.thinking == 'Step by step reasoning...'
        assert mock_post.call_args[1]['json']['reasoning']['effort'] == 'medium'


def test_lmstudio_thinking_returns_content_only(setup_lmstudio_mock):
    """Test that use_thinking=True with return_thinking=False returns only content."""
    mock_response = Mock()
    mock_response.json.return_value = {
        'output': [
            {'type': 'reasoning', 'content': [{'text': 'Thinking...'}]},
            {'type': 'message', 'content': [{'text': 'Final answer.'}]}
        ]
    }
    mock_response.raise_for_status = Mock()

    with patch('LLMFactory.providers.lmstudio.requests.post', return_value=mock_response):
        model = LMStudioInference(model_name="test-model")
        response = model.invoke(
            [{"role": "user", "content": "Question"}],
            "System",
            use_thinking=True,
            return_thinking=False
        )

        assert isinstance(response, str)
        assert response == 'Final answer.'


def test_lmstudio_thinking_effort_levels(setup_lmstudio_mock):
    """Test that effort levels are passed correctly to the API."""
    for effort in ["low", "medium", "high"]:
        mock_response = Mock()
        mock_response.json.return_value = {
            'output': [{'type': 'message', 'content': [{'text': 'OK'}]}]
        }
        mock_response.raise_for_status = Mock()

        with patch('LLMFactory.providers.lmstudio.requests.post', return_value=mock_response) as mock_post:
            model = LMStudioInference(model_name="test-model")
            model.invoke(
                [{"role": "user", "content": "Q"}],
                "S",
                use_thinking=effort,
                return_thinking=True
            )

            assert mock_post.call_args[1]['json']['reasoning']['effort'] == effort


def test_lmstudio_thinking_disabled_uses_sdk(setup_lmstudio_mock, sample_messages, sample_system_prompt):
    """Test that thinking disabled uses SDK instead of REST API."""
    mock_model = Mock()
    mock_model_response = Mock()
    mock_model_response.content = "SDK response"
    mock_model.respond.return_value = mock_model_response
    setup_lmstudio_mock.llm.return_value = mock_model
    setup_lmstudio_mock.Chat.return_value = Mock()

    with patch('LLMFactory.providers.lmstudio.requests.post') as mock_post:
        model = LMStudioInference(model_name="test-model")
        response = model.invoke(sample_messages, sample_system_prompt)

        mock_post.assert_not_called()
        mock_model.respond.assert_called_once()
        assert response == "SDK response"


def test_lmstudio_thinking_rest_api_error(setup_lmstudio_mock):
    """Test that REST API errors are handled gracefully."""
    import requests as req

    with patch('LLMFactory.providers.lmstudio.requests.post') as mock_post:
        mock_post.side_effect = req.RequestException("Connection failed")

        model = LMStudioInference(model_name="test-model")

        with pytest.raises(RuntimeError, match="Error during LM Studio REST API call"):
            model.invoke(
                [{"role": "user", "content": "Q"}],
                "S",
                use_thinking=True
            )


def test_lmstudio_thinking_images_warning(setup_lmstudio_mock, caplog):
    """Test that using images with thinking logs a warning."""
    import logging

    mock_response = Mock()
    mock_response.json.return_value = {
        'output': [{'type': 'message', 'content': [{'text': 'OK'}]}]
    }
    mock_response.raise_for_status = Mock()

    with patch('LLMFactory.providers.lmstudio.requests.post', return_value=mock_response):
        with caplog.at_level(logging.WARNING):
            model = LMStudioInference(model_name="test-model")
            model.invoke(
                [{"role": "user", "content": "Q"}],
                "S",
                use_thinking=True,
                images=["test.jpg"]
            )

            assert "Images are not supported with thinking mode" in caplog.text


def test_lmstudio_thinking_message_formatting(setup_lmstudio_mock):
    """Test that messages are formatted correctly for the REST API."""
    mock_response = Mock()
    mock_response.json.return_value = {
        'output': [{'type': 'message', 'content': [{'text': 'OK'}]}]
    }
    mock_response.raise_for_status = Mock()

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "How are you?"}
    ]

    with patch('LLMFactory.providers.lmstudio.requests.post', return_value=mock_response) as mock_post:
        model = LMStudioInference(model_name="test-model")
        model.invoke(messages, "Be helpful.", use_thinking=True)

        input_text = mock_post.call_args[1]['json']['input']
        assert "System: Be helpful." in input_text
        assert "User: Hello" in input_text
        assert "Assistant: Hi!" in input_text
        assert "User: How are you?" in input_text


def test_lmstudio_thinking_no_reasoning_in_response(setup_lmstudio_mock):
    """Test handling when API returns no reasoning (non-thinking model)."""
    from LLMFactory.llm import ThinkingResponse

    mock_response = Mock()
    mock_response.json.return_value = {
        'output': [
            {'type': 'message', 'content': [{'text': 'Direct response'}]}
        ]
    }
    mock_response.raise_for_status = Mock()

    with patch('LLMFactory.providers.lmstudio.requests.post', return_value=mock_response):
        model = LMStudioInference(model_name="test-model")
        response = model.invoke(
            [{"role": "user", "content": "Q"}],
            "S",
            use_thinking=True,
            return_thinking=True
        )

        assert isinstance(response, ThinkingResponse)
        assert response.content == 'Direct response'
        assert response.thinking is None


# =============================================================================
# Model Loading Status Tests
# =============================================================================

def test_lmstudio_is_model_loaded_true(setup_lmstudio_mock, reset_lmstudio_state):
    """Test is_model_loaded returns True when model is loaded."""
    mock_loaded_model = Mock()
    mock_loaded_model.identifier = "test-model"
    setup_lmstudio_mock.list_loaded_models.return_value = [mock_loaded_model]
    
    model = LMStudioInference(model_name="test-model")
    assert model.is_model_loaded() is True


def test_lmstudio_is_model_loaded_false(setup_lmstudio_mock, reset_lmstudio_state):
    """Test is_model_loaded returns False when model is not loaded."""
    setup_lmstudio_mock.list_loaded_models.return_value = []
    
    model = LMStudioInference(model_name="test-model")
    assert model.is_model_loaded() is False


def test_lmstudio_is_model_loaded_different_model(setup_lmstudio_mock, reset_lmstudio_state):
    """Test is_model_loaded returns False when a different model is loaded."""
    mock_loaded_model = Mock()
    mock_loaded_model.identifier = "other-model"
    setup_lmstudio_mock.list_loaded_models.return_value = [mock_loaded_model]
    
    model = LMStudioInference(model_name="test-model")
    assert model.is_model_loaded() is False


def test_lmstudio_is_model_loaded_exception(setup_lmstudio_mock, reset_lmstudio_state):
    """Test is_model_loaded handles exceptions gracefully."""
    setup_lmstudio_mock.list_loaded_models.side_effect = Exception("API error")
    
    model = LMStudioInference(model_name="test-model")
    assert model.is_model_loaded() is False


def test_lmstudio_verify_connected_and_loaded(setup_lmstudio_mock, reset_lmstudio_state):
    """Test verify returns correct status when model is loaded."""
    mock_loaded_model = Mock()
    mock_loaded_model.identifier = "test-model"
    setup_lmstudio_mock.list_loaded_models.return_value = [mock_loaded_model]
    
    model = LMStudioInference(model_name="test-model")
    result = model.verify()
    
    assert result['connected'] is True
    assert result['model_loaded'] is True
    assert 'test-model' in result['loaded_models']
    assert result['error'] is None


def test_lmstudio_verify_connected_not_loaded(setup_lmstudio_mock, reset_lmstudio_state):
    """Test verify returns correct status when model is not loaded."""
    setup_lmstudio_mock.list_loaded_models.return_value = []
    
    model = LMStudioInference(model_name="test-model")
    result = model.verify()
    
    assert result['connected'] is True
    assert result['model_loaded'] is False
    assert result['loaded_models'] == []
    assert "not loaded" in result['error']


def test_lmstudio_verify_not_connected(setup_lmstudio_mock, reset_lmstudio_state):
    """Test verify returns correct status when server becomes unavailable after init."""
    # Allow initial connection
    setup_lmstudio_mock.Client.is_valid_api_host.return_value = True
    
    model = LMStudioInference(model_name="test-model")
    
    # Simulate server becoming unavailable after model creation
    setup_lmstudio_mock.Client.is_valid_api_host.return_value = False
    
    result = model.verify()
    
    assert result['connected'] is False
    assert "Cannot connect" in result['error']


def test_lmstudio_ensure_model_loaded_already_loaded(setup_lmstudio_mock, reset_lmstudio_state):
    """Test ensure_model_loaded returns True quickly when model is already loaded."""
    mock_loaded_model = Mock()
    mock_loaded_model.identifier = "test-model"
    setup_lmstudio_mock.list_loaded_models.return_value = [mock_loaded_model]
    
    model = LMStudioInference(model_name="test-model")
    result = model.ensure_model_loaded(timeout=1)
    
    assert result is True


def test_lmstudio_clear_model_cache(setup_lmstudio_mock, reset_lmstudio_state):
    """Test _clear_model_cache clears the cached model instance."""
    mock_model = Mock()
    setup_lmstudio_mock.llm.return_value = mock_model
    
    model = LMStudioInference(model_name="test-model")
    model._get_or_load_model()
    
    assert model._model_instance is not None
    model._clear_model_cache()
    assert model._model_instance is None


def test_lmstudio_get_or_load_model_force_reload(setup_lmstudio_mock, reset_lmstudio_state):
    """Test _get_or_load_model with force_reload clears cache."""
    mock_model_1 = Mock()
    mock_model_2 = Mock()
    setup_lmstudio_mock.llm.side_effect = [mock_model_1, mock_model_2]
    
    model = LMStudioInference(model_name="test-model")
    
    # First load
    result1 = model._get_or_load_model()
    assert result1 == mock_model_1
    
    # Force reload should get new model
    result2 = model._get_or_load_model(force_reload=True)
    assert result2 == mock_model_2


# =============================================================================
# Model Not Found Retry Tests
# =============================================================================

def test_lmstudio_invoke_model_not_found_retry(setup_lmstudio_mock, sample_messages, sample_system_prompt, reset_lmstudio_state):
    """Test that invoke retries on LMStudioModelNotFoundError."""
    mock_model = Mock()
    mock_chat = Mock()
    setup_lmstudio_mock.Chat.return_value = mock_chat
    
    # First call raises model not found, second succeeds
    mock_response = Mock()
    mock_response.content = "Success after retry"
    
    # Create a custom exception class that mimics LMStudioModelNotFoundError
    class MockLMStudioModelNotFoundError(Exception):
        pass
    MockLMStudioModelNotFoundError.__name__ = 'LMStudioModelNotFoundError'
    
    mock_model.respond.side_effect = [
        MockLMStudioModelNotFoundError("No model found"),
        mock_response
    ]
    
    # Mock list_loaded_models to return the model on second check
    mock_loaded_model = Mock()
    mock_loaded_model.identifier = "test-model"
    setup_lmstudio_mock.list_loaded_models.side_effect = [
        [],  # First check: not loaded
        [mock_loaded_model],  # Second check: loaded
    ]
    setup_lmstudio_mock.llm.return_value = mock_model
    
    model = LMStudioInference(model_name="test-model")
    
    # Use short wait times for test
    response = model.invoke(
        sample_messages,
        sample_system_prompt,
        _model_reload_retries=2,
        _model_reload_wait_seconds=[0.1, 0.1]
    )
    
    assert response == "Success after retry"
    assert mock_model.respond.call_count == 2


def test_lmstudio_invoke_model_not_found_max_retries(setup_lmstudio_mock, sample_messages, sample_system_prompt, reset_lmstudio_state):
    """Test that invoke fails after max retries on persistent model not found."""
    mock_model = Mock()
    mock_chat = Mock()
    setup_lmstudio_mock.Chat.return_value = mock_chat
    
    # Create exception that matches error detection pattern
    class MockLMStudioModelNotFoundError(Exception):
        pass
    MockLMStudioModelNotFoundError.__name__ = 'LMStudioModelNotFoundError'
    
    mock_model.respond.side_effect = MockLMStudioModelNotFoundError("No model found")
    
    # Model never becomes available
    setup_lmstudio_mock.list_loaded_models.return_value = []
    setup_lmstudio_mock.llm.return_value = mock_model
    
    model = LMStudioInference(model_name="test-model")
    
    with pytest.raises(RuntimeError, match="Error during LM Studio inference"):
        model.invoke(
            sample_messages,
            sample_system_prompt,
            _model_reload_retries=2,
            _model_reload_wait_seconds=[0.1, 0.1]
        )


def test_lmstudio_invoke_non_model_error_no_retry(setup_lmstudio_mock, sample_messages, sample_system_prompt, reset_lmstudio_state):
    """Test that non-model errors don't trigger retry logic."""
    mock_model = Mock()
    mock_chat = Mock()
    setup_lmstudio_mock.Chat.return_value = mock_chat
    
    # Error that is NOT a model-not-found error
    mock_model.respond.side_effect = ValueError("Some other error")
    setup_lmstudio_mock.llm.return_value = mock_model
    
    model = LMStudioInference(model_name="test-model")
    
    with pytest.raises(RuntimeError, match="Error during LM Studio inference"):
        model.invoke(
            sample_messages,
            sample_system_prompt,
            _model_reload_retries=3,
            _model_reload_wait_seconds=[0.1, 0.1, 0.1]
        )
    
    # Should only try once (no retries for non-model errors)
    assert mock_model.respond.call_count == 1


def test_lmstudio_invoke_model_crashed_retry(setup_lmstudio_mock, sample_messages, sample_system_prompt, reset_lmstudio_state):
    """Test that invoke retries on model crashed error."""
    mock_model = Mock()
    mock_chat = Mock()
    setup_lmstudio_mock.Chat.return_value = mock_chat
    
    mock_response = Mock()
    mock_response.content = "Success"
    
    # First call raises model crashed, second succeeds
    mock_model.respond.side_effect = [
        Exception("The model has crashed without additional information"),
        mock_response
    ]
    
    mock_loaded_model = Mock()
    mock_loaded_model.identifier = "test-model"
    setup_lmstudio_mock.list_loaded_models.return_value = [mock_loaded_model]
    setup_lmstudio_mock.llm.return_value = mock_model
    
    model = LMStudioInference(model_name="test-model")
    
    response = model.invoke(
        sample_messages,
        sample_system_prompt,
        _model_reload_retries=2,
        _model_reload_wait_seconds=[0.1, 0.1]
    )
    
    assert response == "Success"
    assert mock_model.respond.call_count == 2
