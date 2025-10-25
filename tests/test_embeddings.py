"""Tests for embedding models."""
import pytest
from unittest.mock import Mock, patch
import numpy as np
from LLMFactory.llm import SentenceTransformerInference, OllamaEmbedInference


@patch('torch.backends.mps.is_available')
@patch('torch.cuda.is_available')
@patch('sentence_transformers.SentenceTransformer')
def test_sentence_transformer_init(mock_st_class, mock_cuda, mock_mps):
    """Test SentenceTransformerInference initialization."""
    mock_cuda.return_value = False
    mock_mps.return_value = False
    mock_st_class.return_value = Mock()

    model = SentenceTransformerInference(model_name="test-model")

    assert model.model_name == "test-model"
    assert model.provider == "sentence-transformer"
    assert model.device == "cpu"


@patch('torch.backends.mps.is_available')
@patch('torch.cuda.is_available')
@patch('sentence_transformers.SentenceTransformer')
def test_sentence_transformer_device_selection_mps(mock_st_class, mock_cuda, mock_mps):
    """Test device selection prefers MPS."""
    mock_cuda.return_value = True
    mock_mps.return_value = True
    mock_st_class.return_value = Mock()

    model = SentenceTransformerInference()

    assert model.device == "mps"


@patch('torch.backends.mps.is_available')
@patch('torch.cuda.is_available')
@patch('sentence_transformers.SentenceTransformer')
def test_sentence_transformer_device_selection_cuda(mock_st_class, mock_cuda, mock_mps):
    """Test device selection uses CUDA when MPS unavailable."""
    mock_cuda.return_value = True
    mock_mps.return_value = False
    mock_st_class.return_value = Mock()

    model = SentenceTransformerInference()

    assert model.device == "cuda"


@patch('torch.backends.mps.is_available')
@patch('torch.cuda.is_available')
@patch('sentence_transformers.SentenceTransformer')
def test_sentence_transformer_invoke_single_string(mock_st_class, mock_cuda, mock_mps):
    """Test invoke with single string."""
    mock_cuda.return_value = False
    mock_mps.return_value = False

    mock_client = Mock()
    mock_embedding = np.array([0.1, 0.2, 0.3])
    mock_client.encode.return_value = np.array([mock_embedding])
    mock_st_class.return_value = mock_client

    model = SentenceTransformerInference()
    result = model.invoke("test text")

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    np.testing.assert_array_equal(result, mock_embedding)


@patch('torch.backends.mps.is_available')
@patch('torch.cuda.is_available')
@patch('sentence_transformers.SentenceTransformer')
def test_sentence_transformer_invoke_list(mock_st_class, mock_cuda, mock_mps):
    """Test invoke with list of strings."""
    mock_cuda.return_value = False
    mock_mps.return_value = False

    mock_client = Mock()
    mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_client.encode.return_value = mock_embeddings
    mock_st_class.return_value = mock_client

    model = SentenceTransformerInference()
    result = model.invoke(["text1", "text2"])

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)


@patch('torch.backends.mps.is_available')
@patch('torch.cuda.is_available')
@patch('sentence_transformers.SentenceTransformer')
def test_sentence_transformer_invoke_with_normalize(mock_st_class, mock_cuda, mock_mps):
    """Test invoke with normalization."""
    mock_cuda.return_value = False
    mock_mps.return_value = False

    mock_client = Mock()
    mock_embedding = np.array([0.1, 0.2, 0.3])
    mock_client.encode.return_value = np.array([mock_embedding])
    mock_st_class.return_value = mock_client

    model = SentenceTransformerInference()
    result = model.invoke("test", normalize=True)

    call_args = mock_client.encode.call_args
    assert call_args[1]['normalize_embeddings'] is True


@patch('ollama.Client')
def test_ollama_embed_init(mock_client_class):
    """Test OllamaEmbedInference initialization."""
    mock_client_class.return_value = Mock()

    model = OllamaEmbedInference(model_name="nomic-embed")

    assert model.model_name == "nomic-embed"
    assert model.provider == "ollama-embed"
    assert model.url == "http://127.0.0.1:11434"


@patch('ollama.Client')
def test_ollama_embed_invoke_single(mock_client_class):
    """Test invoke with single string."""
    mock_client = Mock()
    mock_client.embeddings.return_value = {'embedding': [0.1, 0.2, 0.3]}
    mock_client_class.return_value = mock_client

    model = OllamaEmbedInference()
    result = model.invoke("test text")

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    np.testing.assert_array_almost_equal(result, np.array([0.1, 0.2, 0.3]))


@patch('ollama.Client')
def test_ollama_embed_invoke_list(mock_client_class):
    """Test invoke with list of strings."""
    mock_client = Mock()
    mock_client.embeddings.side_effect = [
        {'embedding': [0.1, 0.2]},
        {'embedding': [0.3, 0.4]}
    ]
    mock_client_class.return_value = mock_client

    model = OllamaEmbedInference()
    result = model.invoke(["text1", "text2"])

    assert isinstance(result, list)
    assert len(result) == 2
    assert mock_client.embeddings.call_count == 2


@patch('ollama.Client')
def test_ollama_embed_invoke_with_normalize(mock_client_class):
    """Test invoke with normalization."""
    mock_client = Mock()
    embedding = [3.0, 4.0]  # Length = 5
    mock_client.embeddings.return_value = {'embedding': embedding}
    mock_client_class.return_value = mock_client

    model = OllamaEmbedInference()
    result = model.invoke("test", normalize=True)

    # Check that result is normalized
    norm = np.linalg.norm(result)
    assert abs(norm - 1.0) < 1e-6


@patch('ollama.Client')
def test_ollama_embed_invoke_to_list(mock_client_class):
    """Test invoke with to_list conversion."""
    mock_client = Mock()
    mock_client.embeddings.return_value = {'embedding': [0.1, 0.2, 0.3]}
    mock_client_class.return_value = mock_client

    model = OllamaEmbedInference()
    result = model.invoke("test", to_list=True)

    assert isinstance(result, list)
    assert len(result) == 3