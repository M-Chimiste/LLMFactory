"""Tests for LMStudioInference using OpenAI-compatible API."""
import pytest
import sys
import threading
from unittest.mock import Mock, patch, MagicMock
from LLMFactory.llm import LMStudioInference, LLMModelFactory
from LLMFactory.providers import lmstudio as lmstudio_module
from LLMFactory import llm as llm_module


@pytest.fixture(autouse=True)
def setup_openai_mock():
    """Setup OpenAI client mock for all tests."""
    mock_openai = Mock()
    mock_client = Mock()
    mock_openai.return_value = mock_client
    
    with patch('LLMFactory.providers.lmstudio.OpenAI', mock_openai):
        # Also mock the requests for connectivity check
        with patch('LLMFactory.providers.lmstudio.requests') as mock_requests:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {"data": []}
            mock_requests.get.return_value = mock_response
            mock_requests.post.return_value = mock_response
            mock_requests.RequestException = Exception
            yield {
                'openai': mock_openai,
                'client': mock_client,
                'requests': mock_requests
            }


@pytest.fixture(autouse=True)
def reset_lmstudio_state():
    """Reset LMStudio cache state before and after each test."""
    llm_module._lmstudio_cache.clear()
    yield
    llm_module._lmstudio_cache.clear()


def test_lmstudio_init_default(setup_openai_mock):
    """Test LMStudioInference initialization with defaults."""
    model = LMStudioInference(model_name="qwen2.5-7b-instruct")

    assert model.model_name == "qwen2.5-7b-instruct"
    assert model.max_new_tokens == 4096
    assert model.temperature == 0.1
    assert model.host == "localhost:1234"
    assert model.provider == "lmstudio"
    assert model.context_length is None
    assert model.gpu_offload is None


def test_lmstudio_init_custom(setup_openai_mock):
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


def test_lmstudio_init_with_env_var(setup_openai_mock, monkeypatch):
    """Test LMStudioInference initialization with environment variable."""
    monkeypatch.setenv("LMSTUDIO_HOST", "remote.server:5678")

    model = LMStudioInference(model_name="test-model")

    assert model.host == "remote.server:5678"


def test_lmstudio_init_connection_error(setup_openai_mock):
    """Test LMStudioInference initialization with connection error."""
    setup_openai_mock['requests'].get.side_effect = Exception("Connection refused")

    with pytest.raises(ConnectionError, match="Cannot connect to LM Studio"):
        LMStudioInference(model_name="test-model")


def test_lmstudio_invoke_basic(setup_openai_mock, sample_messages, sample_system_prompt):
    """Test basic invoke without streaming."""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    setup_openai_mock['client'].chat.completions.create.return_value = mock_response

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(sample_messages, sample_system_prompt)

    assert response == "Test response"
    setup_openai_mock['client'].chat.completions.create.assert_called_once()


def test_lmstudio_invoke_streaming(setup_openai_mock, sample_messages, sample_system_prompt):
    """Test invoke with streaming."""
    # Create mock streaming chunks
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Hello "))]),
        Mock(choices=[Mock(delta=Mock(content="world"))]),
        Mock(choices=[Mock(delta=Mock(content="!"))])
    ]
    setup_openai_mock['client'].chat.completions.create.return_value = iter(mock_chunks)

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(sample_messages, sample_system_prompt, streaming=True)

    result = ''.join(list(response))
    assert result == "Hello world!"


def test_lmstudio_invoke_with_images(setup_openai_mock, sample_messages, sample_system_prompt, sample_image_file):
    """Test invoke with images."""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="I see an image"))]
    setup_openai_mock['client'].chat.completions.create.return_value = mock_response

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(
        sample_messages,
        sample_system_prompt,
        images=[sample_image_file]
    )

    assert response == "I see an image"
    # Verify images were included in the request
    call_args = setup_openai_mock['client'].chat.completions.create.call_args
    messages = call_args[1]['messages']
    last_msg = messages[-1]
    assert isinstance(last_msg['content'], list)
    assert any(c.get('type') == 'image_url' for c in last_msg['content'])


def test_lmstudio_invoke_with_schema(setup_openai_mock, sample_messages, sample_system_prompt):
    """Test invoke with schema."""
    from pydantic import BaseModel

    class TestSchema(BaseModel):
        name: str
        age: int

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='{"name": "John", "age": 30}'))]
    setup_openai_mock['client'].chat.completions.create.return_value = mock_response

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(sample_messages, sample_system_prompt, schema=TestSchema)

    assert response == '{"name": "John", "age": 30}'
    # Verify schema was included in the request
    call_args = setup_openai_mock['client'].chat.completions.create.call_args
    assert 'response_format' in call_args[1]
    assert call_args[1]['response_format']['type'] == 'json_schema'


def test_lmstudio_invoke_with_custom_params(setup_openai_mock, sample_messages, sample_system_prompt):
    """Test invoke with custom parameters."""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    setup_openai_mock['client'].chat.completions.create.return_value = mock_response

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(
        sample_messages,
        sample_system_prompt,
        max_tokens=1024,
        temperature=0.9,
        top_p=0.95,
        seed=42
    )

    assert response == "Test response"
    call_args = setup_openai_mock['client'].chat.completions.create.call_args[1]
    assert call_args['temperature'] == 0.9
    assert call_args['max_tokens'] == 1024
    assert call_args['top_p'] == 0.95
    assert call_args['seed'] == 42


def test_lmstudio_invoke_error_handling(setup_openai_mock, sample_messages, sample_system_prompt):
    """Test invoke error handling."""
    setup_openai_mock['client'].chat.completions.create.side_effect = Exception("API error")

    model = LMStudioInference(model_name="test-model")
    with pytest.raises(RuntimeError, match="Error during LM Studio inference"):
        model.invoke(sample_messages, sample_system_prompt)


# =============================================================================
# Model Management Tests
# =============================================================================

def test_lmstudio_is_model_loaded_true(setup_openai_mock):
    """Test is_model_loaded returns True when model is loaded."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "models": [{
            "key": "test-model",
            "loaded_instances": [{"id": "test-model-instance"}]
        }]
    }
    mock_response.raise_for_status = Mock()
    setup_openai_mock['requests'].get.return_value = mock_response

    model = LMStudioInference(model_name="test-model")
    assert model.is_model_loaded() is True


def test_lmstudio_is_model_loaded_false(setup_openai_mock):
    """Test is_model_loaded returns False when model is not loaded."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "models": [{
            "key": "other-model",
            "loaded_instances": []
        }]
    }
    mock_response.raise_for_status = Mock()
    setup_openai_mock['requests'].get.return_value = mock_response

    model = LMStudioInference(model_name="test-model")
    assert model.is_model_loaded() is False


def test_lmstudio_get_loaded_models(setup_openai_mock):
    """Test get_loaded_models returns list of loaded models."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "models": [
            {
                "key": "model-a",
                "display_name": "Model A",
                "type": "llm",
                "loaded_instances": [{"id": "model-a-1"}]
            },
            {
                "key": "model-b",
                "display_name": "Model B",
                "type": "llm",
                "loaded_instances": []
            }
        ]
    }
    mock_response.raise_for_status = Mock()
    setup_openai_mock['requests'].get.return_value = mock_response

    model = LMStudioInference(model_name="test-model")
    loaded = model.get_loaded_models()

    assert len(loaded) == 1
    assert loaded[0]['key'] == 'model-a'


def test_lmstudio_load_model_explicit(setup_openai_mock):
    """Test explicit model loading."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "status": "loaded",
        "load_time_seconds": 5.5
    }
    mock_response.raise_for_status = Mock()
    setup_openai_mock['requests'].post.return_value = mock_response

    model = LMStudioInference(model_name="test-model", context_length=8192)
    result = model.load_model_explicit()

    assert result is True
    # Verify the load request was made with correct params
    call_args = setup_openai_mock['requests'].post.call_args
    assert 'test-model' in str(call_args)


def test_lmstudio_unload_model(setup_openai_mock):
    """Test model unloading."""
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    setup_openai_mock['requests'].post.return_value = mock_response

    model = LMStudioInference(model_name="test-model")
    result = model.unload_model()

    assert result is True


def test_lmstudio_verify_connected_and_loaded(setup_openai_mock):
    """Test verify returns correct status when connected and model loaded."""
    # Mock the models endpoint response
    mock_api_response = Mock()
    mock_api_response.json.return_value = {
        "models": [{
            "key": "test-model",
            "loaded_instances": [{"id": "test-model"}]
        }]
    }
    mock_api_response.raise_for_status = Mock()
    
    # Return different mocks for different URLs
    def mock_get(url, **kwargs):
        return mock_api_response
    
    setup_openai_mock['requests'].get.side_effect = mock_get

    model = LMStudioInference(model_name="test-model")
    result = model.verify()

    assert result['connected'] is True
    assert result['model_loaded'] is True
    assert result['error'] is None


def test_lmstudio_verify_not_loaded(setup_openai_mock):
    """Test verify returns correct status when model not loaded."""
    mock_api_response = Mock()
    mock_api_response.json.return_value = {
        "models": [{
            "key": "other-model",
            "loaded_instances": []
        }]
    }
    mock_api_response.raise_for_status = Mock()
    setup_openai_mock['requests'].get.return_value = mock_api_response

    model = LMStudioInference(model_name="test-model")
    result = model.verify()

    assert result['connected'] is True
    assert result['model_loaded'] is False
    assert "not loaded" in result['error']


# =============================================================================
# Retry Logic Tests
# =============================================================================

def test_lmstudio_invoke_model_not_found_retry(setup_openai_mock, sample_messages, sample_system_prompt):
    """Test that invoke retries on model not found error."""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Success"))]
    
    # First call fails, second succeeds
    setup_openai_mock['client'].chat.completions.create.side_effect = [
        Exception("No model found"),
        mock_response
    ]
    
    # Mock is_model_loaded and load_model_explicit
    mock_api_response = Mock()
    mock_api_response.json.return_value = {"models": [{"key": "test-model", "loaded_instances": [{"id": "test"}]}]}
    mock_api_response.raise_for_status = Mock()
    setup_openai_mock['requests'].get.return_value = mock_api_response
    setup_openai_mock['requests'].post.return_value = mock_api_response

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(
        sample_messages,
        sample_system_prompt,
        _model_reload_retries=2,
        _model_reload_wait_seconds=[0.1]
    )

    assert response == "Success"
    assert setup_openai_mock['client'].chat.completions.create.call_count == 2


def test_lmstudio_invoke_max_retries_exceeded(setup_openai_mock, sample_messages, sample_system_prompt):
    """Test that invoke fails after max retries."""
    setup_openai_mock['client'].chat.completions.create.side_effect = Exception("No model found")
    
    mock_api_response = Mock()
    mock_api_response.json.return_value = {"models": []}
    mock_api_response.raise_for_status = Mock()
    setup_openai_mock['requests'].get.return_value = mock_api_response
    setup_openai_mock['requests'].post.return_value = mock_api_response

    model = LMStudioInference(model_name="test-model")
    
    with pytest.raises(RuntimeError, match="Error during LM Studio inference"):
        model.invoke(
            sample_messages,
            sample_system_prompt,
            _model_reload_retries=2,
            _model_reload_wait_seconds=[0.1]
        )


def test_lmstudio_invoke_non_model_error_no_retry(setup_openai_mock, sample_messages, sample_system_prompt):
    """Test that non-model errors don't trigger retry."""
    setup_openai_mock['client'].chat.completions.create.side_effect = ValueError("Some other error")

    model = LMStudioInference(model_name="test-model")
    
    with pytest.raises(RuntimeError, match="Error during LM Studio inference"):
        model.invoke(
            sample_messages,
            sample_system_prompt,
            _model_reload_retries=3,
            _model_reload_wait_seconds=[0.1]
        )
    
    # Should only try once
    assert setup_openai_mock['client'].chat.completions.create.call_count == 1


# =============================================================================
# Thinking Mode Tests
# =============================================================================

def test_lmstudio_thinking_returns_thinking_response(setup_openai_mock):
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
    setup_openai_mock['requests'].post.return_value = mock_response

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


def test_lmstudio_thinking_returns_content_only(setup_openai_mock):
    """Test that use_thinking=True with return_thinking=False returns only content."""
    mock_response = Mock()
    mock_response.json.return_value = {
        'output': [
            {'type': 'reasoning', 'content': [{'text': 'Thinking...'}]},
            {'type': 'message', 'content': [{'text': 'Final answer.'}]}
        ]
    }
    mock_response.raise_for_status = Mock()
    setup_openai_mock['requests'].post.return_value = mock_response

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(
        [{"role": "user", "content": "Question"}],
        "System",
        use_thinking=True,
        return_thinking=False
    )

    assert isinstance(response, str)
    assert response == 'Final answer.'


def test_lmstudio_thinking_effort_levels(setup_openai_mock):
    """Test that effort levels are passed correctly to the API."""
    mock_response = Mock()
    mock_response.json.return_value = {
        'output': [{'type': 'message', 'content': [{'text': 'OK'}]}]
    }
    mock_response.raise_for_status = Mock()
    setup_openai_mock['requests'].post.return_value = mock_response

    for effort in ["low", "medium", "high"]:
        model = LMStudioInference(model_name="test-model")
        model.invoke(
            [{"role": "user", "content": "Q"}],
            "S",
            use_thinking=effort,
            return_thinking=True
        )

        call_args = setup_openai_mock['requests'].post.call_args
        assert call_args[1]['json']['reasoning']['effort'] == effort


def test_lmstudio_thinking_disabled_uses_openai_client(setup_openai_mock, sample_messages, sample_system_prompt):
    """Test that thinking disabled uses OpenAI client instead of REST API."""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="OpenAI response"))]
    setup_openai_mock['client'].chat.completions.create.return_value = mock_response

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(sample_messages, sample_system_prompt)

    setup_openai_mock['client'].chat.completions.create.assert_called_once()
    assert response == "OpenAI response"


def test_lmstudio_thinking_images_warning(setup_openai_mock, caplog):
    """Test that using images with thinking logs a warning."""
    import logging

    mock_response = Mock()
    mock_response.json.return_value = {
        'output': [{'type': 'message', 'content': [{'text': 'OK'}]}]
    }
    mock_response.raise_for_status = Mock()
    setup_openai_mock['requests'].post.return_value = mock_response

    with caplog.at_level(logging.WARNING):
        model = LMStudioInference(model_name="test-model")
        model.invoke(
            [{"role": "user", "content": "Q"}],
            "S",
            use_thinking=True,
            images=["test.jpg"]
        )

        assert "Images not supported with thinking mode" in caplog.text


# =============================================================================
# Factory and Caching Tests
# =============================================================================

def test_lmstudio_factory_creates_instance(setup_openai_mock):
    """Test that factory creates LMStudio instance correctly."""
    model = LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='test-model',
        host='localhost:1234'
    )
    
    assert model is not None
    assert model.model_name == 'test-model'
    assert model.provider == 'lmstudio'


def test_lmstudio_factory_caches_instance(setup_openai_mock):
    """Test that factory caches LMStudio instances."""
    model1 = LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='test-model',
        host='localhost:1234'
    )
    
    model2 = LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='test-model',
        host='localhost:1234'
    )
    
    assert model1 is model2


def test_lmstudio_factory_different_models_different_instances(setup_openai_mock):
    """Test that different models get different instances."""
    model1 = LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='model-a',
        host='localhost:1234'
    )
    
    model2 = LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='model-b',
        host='localhost:1234'
    )
    
    assert model1 is not model2


def test_lmstudio_clear_cache(setup_openai_mock):
    """Test cache clearing functionality."""
    model1 = LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='test-model',
        host='localhost:1234'
    )
    
    LLMModelFactory.clear_lmstudio_cache()
    
    model2 = LLMModelFactory.create_model(
        model_type='lmstudio',
        model_name='test-model',
        host='localhost:1234'
    )
    
    assert model1 is not model2


def test_lmstudio_close_noop(setup_openai_mock):
    """Test that close is a no-op for HTTP client."""
    model = LMStudioInference(model_name="test-model")
    # Should not raise
    model.close()


# =============================================================================
# Edge Cases
# =============================================================================

def test_lmstudio_host_with_protocol(setup_openai_mock):
    """Test that host with protocol is handled correctly."""
    model = LMStudioInference(model_name="test-model", host="http://myserver:1234")
    
    assert model.base_url == "http://myserver:1234"
    assert model.host == "myserver:1234"


def test_lmstudio_streaming_empty_chunks(setup_openai_mock, sample_messages, sample_system_prompt):
    """Test streaming handles empty chunks correctly."""
    mock_chunks = [
        Mock(choices=[Mock(delta=Mock(content="Hello"))]),
        Mock(choices=[Mock(delta=Mock(content=None))]),  # Empty chunk
        Mock(choices=[Mock(delta=Mock(content=" world"))]),
    ]
    setup_openai_mock['client'].chat.completions.create.return_value = iter(mock_chunks)

    model = LMStudioInference(model_name="test-model")
    response = model.invoke(sample_messages, sample_system_prompt, streaming=True)

    result = ''.join(list(response))
    assert result == "Hello world"
