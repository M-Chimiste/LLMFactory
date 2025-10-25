# LLMFactory

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A unified factory pattern interface for multiple LLM inference providers with multimodal support. LLMFactory simplifies working with different LLM APIs by providing a consistent interface across providers.

## Features

- **Unified Interface**: Single API for multiple LLM providers
- **Multimodal Support**: Built-in image processing for vision-capable models
- **Streaming Support**: Token-by-token streaming for all providers
- **Schema Support**: Structured JSON output with Pydantic models
- **Flexible Configuration**: Environment variables, direct parameters, or config files
- **Type Safety**: Full type hints for better IDE support

## Supported Providers

### Chat/Completion Models
- **Ollama** - Local model inference
- **Anthropic** - Claude models via API
- **Anthropic Bedrock** - Claude models via AWS Bedrock
- **OpenAI** - GPT models
- **Google Gemini** - Gemini models
- **Llama.cpp** - Local GGUF models
- **Custom OpenAI-Compatible** - Any OpenAI-compatible API

### Embedding Models
- **Sentence Transformers** - HuggingFace models
- **Ollama Embeddings** - Local Ollama embedding models

## Installation


### From Source
```bash
git clone https://github.com/M-Chimiste/LLMFactory.git
cd LLMFactory
pip install -e .
```

### Development Installation
```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from LLMFactory import LLMModelFactory

# Create a model instance
model = LLMModelFactory.create_model(
    model_type='ollama',
    model_name='llama3',
    temperature=0.7
)

# Generate a response
response = model.invoke(
    messages=[{"role": "user", "content": "What is Python?"}],
    system_prompt="You are a helpful programming assistant."
)

print(response)
```

### Streaming Responses

```python
model = LLMModelFactory.create_model(
    model_type='anthropic',
    model_name='claude-3-sonnet-20240229'
)

response_stream = model.invoke(
    messages=[{"role": "user", "content": "Write a short poem."}],
    system_prompt="You are a creative poet.",
    streaming=True
)

for chunk in response_stream:
    print(chunk, end='', flush=True)
```

### Multimodal (Vision) Inference

```python
model = LLMModelFactory.create_model(
    model_type='anthropic',
    model_name='claude-3-sonnet-20240229'
)

response = model.invoke(
    messages=[{"role": "user", "content": "What's in this image?"}],
    system_prompt="You are a helpful vision assistant.",
    images=["path/to/image.jpg"]  # Can also use bytes
)

print(response)
```

### Structured Output with Schema

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

model = LLMModelFactory.create_model(
    model_type='openai',
    model_name='gpt-4'
)

response = model.invoke(
    messages=[{"role": "user", "content": "Tell me about Albert Einstein"}],
    system_prompt="Extract person information.",
    schema=Person
)

print(response)  # JSON formatted according to Person schema
```

## Provider-Specific Examples

### Ollama

```python
model = LLMModelFactory.create_model(
    model_type='ollama',
    model_name='llama3',
    url='http://localhost:11434',
    num_ctx=8192,
    temperature=0.7
)
```

### Anthropic

```python
import os
os.environ['ANTHROPIC_API_KEY'] = 'your-api-key'

model = LLMModelFactory.create_model(
    model_type='anthropic',
    model_name='claude-3-opus-20240229',
    max_new_tokens=4096
)
```

### Anthropic Bedrock (AWS)

```python
# Option 1: Using AWS profile
model = LLMModelFactory.create_model(
    model_type='anthropic-bedrock',
    model_name='anthropic.claude-3-sonnet-20240229-v1:0',
    aws_profile='my-profile',
    region_name='us-west-2'
)

# Option 2: Using environment variables
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
model = LLMModelFactory.create_model(
    model_type='anthropic-bedrock',
    model_name='anthropic.claude-3-haiku-20240307-v1:0'
)

# Option 3: Using EC2 instance role (credentials auto-detected)
model = LLMModelFactory.create_model(
    model_type='anthropic-bedrock'
)
```

### OpenAI

```python
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key'

model = LLMModelFactory.create_model(
    model_type='openai',
    model_name='gpt-4-turbo',
    temperature=0.8
)
```

### Google Gemini

```python
import os
os.environ['GOOGLE_API_KEY'] = 'your-api-key'

model = LLMModelFactory.create_model(
    model_type='gemini',
    model_name='gemini-1.5-pro',
    max_new_tokens=8192
)
```

### Llama.cpp (Local GGUF)

```python
model = LLMModelFactory.create_model(
    model_type='llamacpp',
    model_name='/path/to/model.gguf',
    n_gpu_layers=35,
    num_ctx=4096
)
```

### Custom OpenAI-Compatible API

```python
model = LLMModelFactory.create_model(
    model_type='custom_oai',
    model_name='custom-model',
    base_url='http://localhost:8000/v1',
    api_key='optional-key'
)
```

## Embedding Models

### Sentence Transformers

```python
model = LLMModelFactory.create_model(
    model_type='sentence-transformer',
    model_name='Alibaba-NLP/gte-large-en-v1.5'
)

# Single text
embedding = model.invoke("Hello world")

# Multiple texts
embeddings = model.invoke(["Text 1", "Text 2"], normalize=True)
```

### Ollama Embeddings

```python
model = LLMModelFactory.create_model(
    model_type='ollama-embed',
    model_name='nomic-embed-text'
)

embedding = model.invoke("Hello world", normalize=True)
```

## Environment Variables

LLMFactory uses environment variables for API keys and configuration:

```bash
# Anthropic
export ANTHROPIC_API_KEY=your-key

# OpenAI
export OPENAI_API_KEY=your-key

# Google Gemini
export GOOGLE_API_KEY=your-key

# Ollama
export OLLAMA_URL=http://localhost:11434

# AWS Bedrock
export AWS_PROFILE=your-profile
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret

# Custom OpenAI-Compatible
export CUSTOM_OAI_BASE_URL=http://localhost:8000
export CUSTOM_OAI_API_KEY=your-key
```

## Advanced Features

### Message History

```python
messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
    {"role": "user", "content": "What about 3+3?"}
]

response = model.invoke(
    messages=messages,
    system_prompt="You are a math tutor."
)
```

### Multiple Images

```python
response = model.invoke(
    messages=[{"role": "user", "content": "Compare these images"}],
    system_prompt="You are a vision assistant.",
    images=["image1.jpg", "image2.jpg"]
)
```

### Custom Model Parameters

Each provider supports specific parameters:

```python
# Ollama with custom context
model = LLMModelFactory.create_model(
    model_type='ollama',
    model_name='llama3',
    num_ctx=32768,
    request_timeout=120.0
)

# Llama.cpp with GPU layers
model = LLMModelFactory.create_model(
    model_type='llamacpp',
    model_name='model.gguf',
    n_gpu_layers=40,
    n_ctx=8192,
    verbose=True
)
```

## Architecture

LLMFactory uses an abstract factory pattern:

```
InferenceModel (ABC)
├── OllamaInference
├── AnthropicInference
├── AnthropicBedrockInference
├── OpenAIInference
├── GeminiInference
├── LlamacppInference
├── CustomOAIInference
├── SentenceTransformerInference
└── OllamaEmbedInference

LLMModelFactory.create_model() -> InferenceModel
```

All models implement:
- `invoke()` - Generate text/embeddings
- `_load_model()` - Initialize the provider client
- `_get_provider()` - Return provider name

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=LLMFactory --cov-report=html

# Run specific test file
pytest tests/test_ollama.py -v
```

### Code Style

```bash
# Format code
black LLMFactory/

# Lint
flake8 LLMFactory/

# Type check
mypy LLMFactory/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of excellent provider SDKs: anthropic, openai, google-generativeai, ollama, llama-cpp-python
- Inspired by the need for a unified LLM interface

## Support

- GitHub Issues: [https://github.com/M-Chimiste/LLMFactory/issues](https://github.com/M-Chimiste/LLMFactory/issues)
- Documentation: [https://github.com/M-Chimiste/LLMFactory](https://github.com/M-Chimiste/LLMFactory)

## Changelog

### v0.0.1 (Initial Release)
- Unified interface for 8 LLM providers
- Multimodal support for vision models
- Streaming support
- Schema-based structured output
- Embedding model support
- AWS Bedrock integration with flexible credential management