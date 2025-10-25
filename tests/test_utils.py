"""Tests for utility functions."""
import pytest
import base64
from LLMFactory.llm import _encode_image


def test_encode_image_from_bytes(sample_image_bytes):
    """Test encoding image from bytes."""
    result = _encode_image(sample_image_bytes)

    assert isinstance(result, str)
    # Verify it's valid base64
    decoded = base64.b64decode(result)
    assert decoded == sample_image_bytes


def test_encode_image_from_file(sample_image_file):
    """Test encoding image from file path."""
    result = _encode_image(sample_image_file)

    assert isinstance(result, str)
    # Verify it's valid base64
    decoded = base64.b64decode(result)
    assert len(decoded) > 0


def test_encode_image_file_not_found():
    """Test encoding image with non-existent file."""
    with pytest.raises(FileNotFoundError):
        _encode_image("/nonexistent/path/image.jpg")