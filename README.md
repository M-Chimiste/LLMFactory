# LLMFactory

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A unified factory pattern interface for multiple LLM inference providers with multimodal support. LLMFactory simplifies working with different LLM APIs by providing a consistent interface across providers.

## Features

- **Unified Interface**: Single API for multiple LLM providers
- **Multimodal Support**: Built-in image processing for vision-capable models
- **Streaming Support**: Token-by-token streaming for all providers
- **Thinking/Reasoning Mode**: Access model reasoning traces (Ollama, LM Studio)
- **Schema Support**: Structured JSON output with Pydantic models
- **Flexible Configuration**: Environment variables, direct parameters, or config files
- **Type Safety**: Full type hints for better IDE support

## Supported Providers

### Chat/Completion Models
- **Ollama** - Local model inference
- **LM Studio** - Local/remote inference with advanced configuration (GPU offload, large context)
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

#### Ollama Thinking Mode

```python
from LLMFactory.llm import OllamaInference

model = OllamaInference(model_name="qwen3")

# Enable thinking and return both thinking trace and answer
response = model.invoke(
    messages=[{"role": "user", "content": "What is 25 * 4?"}],
    system_prompt="You are a helpful assistant.",
    use_thinking=True,
    return_thinking=True
)

print(f"Thinking: {response.thinking}")
print(f"Answer: {response.content}")

# For GPT-OSS models, use effort levels
response = model.invoke(
    messages=[{"role": "user", "content": "Explain quantum entanglement"}],
    system_prompt="You are a physics teacher.",
    use_thinking="high",  # "low", "medium", or "high"
    return_thinking=True
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

### LM Studio

LM Studio provides a desktop app and local server for running LLMs. The `lmstudio` provider supports both local and remote connections with advanced configuration.

> **Note**: The LMStudio SDK uses a singleton pattern for its default client. LLMFactory handles this **automatically and transparently** - you can create multiple model instances without any special handling. Just use `LLMModelFactory.create_model()` as you would with any other provider.

#### Basic Local Usage

```python
# Local connection (default: localhost:1234)
model = LLMModelFactory.create_model(
    model_type='lmstudio',
    model_name='qwen2.5-7b-instruct',
    temperature=0.7
)

response = model.invoke(
    messages=[{"role": "user", "content": "Hello!"}],
    system_prompt="You are a helpful assistant."
)
```

#### Remote Connection

```python
# Connect to remote LM Studio instance
model = LLMModelFactory.create_model(
    model_type='lmstudio',
    model_name='llama-3.1-8b',
    host='athena.local:1234',  # or use IP: '192.168.1.100:1234'
    temperature=0.7
)

# Or use environment variable
# export LMSTUDIO_HOST=athena.local:1234
```

#### Advanced Configuration

```python
# Full GPU offload with large context window
model = LLMModelFactory.create_model(
    model_type='lmstudio',
    model_name='qwen2.5-7b-instruct',
    context_length=32768,      # Set context window size
    gpu_offload='max',          # Options: 'max', 'off', or float 0-1
    temperature=0.8
)

# Partial GPU offload (50% of layers on GPU)
model = LLMModelFactory.create_model(
    model_type='lmstudio',
    model_name='mistral-7b',
    context_length=16384,
    gpu_offload=0.5,            # 50% of layers on GPU
    max_new_tokens=2048
)
```

#### Structured Output with Schema

```python
from pydantic import BaseModel

class BookInfo(BaseModel):
    title: str
    author: str
    year: int

model = LLMModelFactory.create_model(
    model_type='lmstudio',
    model_name='qwen2.5-7b-instruct'
)

# Non-streaming structured output
book = model.invoke(
    messages=[{"role": "user", "content": "Tell me about The Hobbit"}],
    system_prompt="Extract book information.",
    schema=BookInfo
)

print(f"Title: {book.title}, Author: {book.author}, Year: {book.year}")

# Streaming with structured output
response_stream = model.invoke(
    messages=[{"role": "user", "content": "Tell me about 1984"}],
    system_prompt="Extract book information.",
    schema=BookInfo,
    streaming=True
)

# Stream tokens as they arrive
for chunk in response_stream:
    print(chunk, end='', flush=True)
```

#### Vision Models (Multimodal)

```python
# Use LM Studio with vision-capable models
model = LLMModelFactory.create_model(
    model_type='lmstudio',
    model_name='llava-1.5-7b'
)

response = model.invoke(
    messages=[{"role": "user", "content": "What's in this image?"}],
    system_prompt="You are a vision assistant.",
    images=["photo.jpg"]  # Supports file paths or bytes
)
```

#### Sampling Parameters

```python
response = model.invoke(
    messages=[{"role": "user", "content": "Write a creative story"}],
    system_prompt="You are a creative writer.",
    max_tokens=1024,
    temperature=0.9,
    top_p=0.95,
    top_k=50
)
```

#### Thinking/Reasoning Mode

LM Studio supports thinking mode for models with reasoning capabilities (e.g., `zai-org/glm-4.7-flash`). When enabled, the model shows its reasoning process before providing an answer.

```python
from LLMFactory.llm import LMStudioInference

model = LMStudioInference(model_name="zai-org/glm-4.7-flash")

# Enable thinking with effort level
response = model.invoke(
    messages=[{"role": "user", "content": "What is the derivative of x^2?"}],
    system_prompt="You are a calculus tutor.",
    use_thinking="medium",  # "low", "medium", or "high"
    return_thinking=True
)

print(f"Reasoning: {response.thinking}")
print(f"Answer: {response.content}")

# Models without thinking support still work (thinking will be None)
model2 = LMStudioInference(model_name="granite-4.0-h-tiny-mlx")
response2 = model2.invoke(
    messages=[{"role": "user", "content": "Hello"}],
    system_prompt="Be helpful.",
    use_thinking=True,
    return_thinking=True
)
print(f"Thinking: {response2.thinking}")  # None for non-thinking models
print(f"Content: {response2.content}")
```

> **Note**: When `use_thinking` is enabled, LM Studio uses the `/v1/responses` REST endpoint instead of the SDK. Some features like images and schema are not available in thinking mode.

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

# LM Studio
export LMSTUDIO_HOST=localhost:1234  # or remote: athena.local:1234

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

### Thinking/Reasoning Mode

Some models support "thinking" or "reasoning" mode, where they show their step-by-step thought process before providing an answer. LLMFactory provides a unified interface for this feature across supported providers (Ollama, LM Studio).

```python
from LLMFactory.llm import OllamaInference, ThinkingResponse

# Enable thinking mode and get both thinking trace and content
model = OllamaInference(model_name="qwen3")
response = model.invoke(
    messages=[{"role": "user", "content": "What is 15 * 23?"}],
    system_prompt="You are a math tutor.",
    use_thinking=True,
    return_thinking=True
)

# Response is a ThinkingResponse object
print(f"Thinking: {response.thinking}")
print(f"Answer: {response.content}")
```

**Parameters:**
- `use_thinking`: Enable thinking mode. Can be:
  - `True`/`False` - Enable/disable thinking
  - `"low"`, `"medium"`, `"high"` - Effort level (provider-specific)
- `return_thinking`: If `True`, return a `ThinkingResponse` with both `thinking` and `content` fields. If `False` (default), return only the content string.

**Provider-Specific Behavior:**

| Provider | Parameter | API Used |
|----------|-----------|----------|
| Ollama | `think=True/False` or `"low"/"medium"/"high"` | Native `think` parameter |
| LM Studio | `reasoning.effort: "low"/"medium"/"high"` | REST `/v1/responses` endpoint |

**Example with LM Studio:**

```python
from LLMFactory.llm import LMStudioInference

model = LMStudioInference(model_name="zai-org/glm-4.7-flash")

# High effort reasoning
response = model.invoke(
    messages=[{"role": "user", "content": "Solve: If a train travels 120km in 2 hours, what is its speed?"}],
    system_prompt="You are a physics tutor.",
    use_thinking="high",
    return_thinking=True
)

print(f"Reasoning: {response.thinking}")
print(f"Answer: {response.content}")
```

**Notes:**
- Models that don't support thinking will return `thinking=None` even when `use_thinking=True`
- Streaming with thinking is supported - thinking tokens are prefixed with `[THINKING]`
- When `return_thinking=False` (default), only content is returned for backward compatibility

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

### Modular Provider Design (v0.2.0+)

LLMFactory features a clean, modular architecture with providers organized in dedicated modules:

```
LLMFactory/
├── llm.py                    # Factory and main exports
└── providers/
    ├── base.py              # InferenceModel ABC + utilities
    ├── ollama.py            # Ollama providers
    ├── lmstudio.py          # LM Studio provider
    ├── anthropic.py         # Anthropic providers
    ├── openai.py            # OpenAI providers
    ├── gemini.py            # Google Gemini provider
    ├── embeddings.py        # Sentence Transformers
    └── llamacpp.py          # llama.cpp provider
```

**Class Hierarchy:**
```
InferenceModel (ABC)
├── OllamaInference
├── LMStudioInference
├── AnthropicInference
├── AnthropicBedrockInference
├── OpenAIInference
├── CustomOAIInference
├── GeminiInference
├── LlamacppInference
├── SentenceTransformerInference
└── OllamaEmbedInference

LLMModelFactory.create_model() -> InferenceModel
```

**All models implement:**
- `invoke()` - Generate text/embeddings
- `_load_model()` - Initialize the provider client
- `_get_provider()` - Return provider name

**Import Flexibility:**
```python
# Import from main module (recommended for compatibility)
from LLMFactory.llm import OllamaInference, LLMModelFactory, ThinkingResponse

# Import from specific provider module
from LLMFactory.providers.ollama import OllamaInference
from LLMFactory.providers import AnthropicInference, ThinkingResponse
```

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

### Adding a New Provider

LLMFactory's modular architecture makes it easy to add new providers:

1. **Create provider module**: `LLMFactory/providers/your_provider.py`
   ```python
   from .base import InferenceModel

   class YourProviderInference(InferenceModel):
       def _get_provider(self) -> str:
           return "your-provider"

       def _load_model(self):
           # Initialize your provider's client
           pass

       def invoke(self, messages, system_prompt, **kwargs):
           # Implement inference logic
           pass
   ```

2. **Export provider**: Add to `LLMFactory/providers/__init__.py`
   ```python
   from .your_provider import YourProviderInference

   __all__ = [..., 'YourProviderInference']
   ```

3. **Register in factory**: Update `LLMFactory/llm.py`
   ```python
   from .providers import (..., YourProviderInference)

   class LLMModelFactory:
       _models = {
           ...,
           'your-provider': YourProviderInference
       }
   ```

4. **Add tests**: Create `tests/test_your_provider.py`

See [RELEASE.md](RELEASE.md) for more details on the modular architecture.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow the existing code style and structure
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR
- Keep commits focused and atomic

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of excellent provider SDKs: anthropic, openai, google-generativeai, ollama, lmstudio-python, llama-cpp-python
- Inspired by the need for a unified LLM interface

## Support

- GitHub Issues: [https://github.com/M-Chimiste/LLMFactory/issues](https://github.com/M-Chimiste/LLMFactory/issues)
- Documentation: [https://github.com/M-Chimiste/LLMFactory](https://github.com/M-Chimiste/LLMFactory)

## Changelog

### v0.2.5 (Thinking/Reasoning Mode)
- **🧠 Thinking Mode**: Added `use_thinking` and `return_thinking` parameters for Ollama and LM Studio
- **📦 ThinkingResponse**: New dataclass containing both `thinking` (reasoning trace) and `content` (final answer)
- **🔧 Ollama**: Uses native `think` parameter, supports `True`/`False` or `"low"`/`"medium"`/`"high"` effort levels
- **🔧 LM Studio**: Uses `/v1/responses` REST endpoint with `reasoning.effort` parameter
- **📡 Streaming Support**: Thinking mode works with streaming (thinking tokens prefixed with `[THINKING]`)
- **✅ Graceful Fallback**: Non-thinking models return `thinking=None` when thinking is requested

### v0.2.1 (LM Studio Singleton Handling)
- **🔧 LMStudio Fix**: Resolved `LMStudioClientError: Default client is already created` error
- **♻️ Transparent Caching**: Multiple LMStudio instances now work seamlessly without any API changes
- **🔒 Thread Safety**: Added thread-safe instance caching for concurrent usage
- **⚠️ Host Warning**: Clear warning when attempting to use different hosts (SDK limitation)

### v0.2.0 (Modular Provider Architecture)
- **🏗️ Major Refactoring**: Reorganized codebase into modular provider architecture
- **📦 Improved Maintainability**: Each provider now in dedicated module under `LLMFactory/providers/`
- **✅ 100% Backward Compatible**: All existing imports continue to work
- **🚀 Developer Experience**: Simplified process for adding new providers
- **📊 Code Reduction**: Main `llm.py` reduced from 1,269 to 109 lines (91% reduction)
- **✨ Import Flexibility**: Can import from main module or specific provider modules
- **🔧 Better Organization**: Clear separation of concerns with `base.py`, provider-specific modules
- See [RELEASE.md](RELEASE.md) for detailed migration guide and architecture overview

### v0.1.0 (LM Studio Support)
- Added LM Studio provider with remote connection support
- Enhanced multimodal capabilities
- Improved test coverage

### v0.0.1 (Initial Release)
- Unified interface for 8 LLM providers
- Multimodal support for vision models
- Streaming support
- Schema-based structured output
- Embedding model support
- AWS Bedrock integration with flexible credential management