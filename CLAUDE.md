# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLMFactory is a Python library providing a unified factory-pattern interface for multiple LLM inference providers (Ollama, LM Studio, Anthropic, OpenAI, Gemini, llama.cpp, and more). Python 3.11+ required.

## Commands

```bash
# Install (editable with dev deps)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_ollama.py -v

# Run tests by marker
pytest -m unit
pytest -m "not slow"

# Run with coverage
pytest --cov=LLMFactory --cov-report=html

# Format
black LLMFactory/

# Lint
flake8 LLMFactory/

# Type check
mypy LLMFactory/
```

## Architecture

**Factory pattern entry point:** `LLMModelFactory.create_model(model_type, **kwargs)` in `LLMFactory/llm.py` dispatches to provider classes via a `_models` dict. It uses `inspect.signature` to filter kwargs so only valid params reach each provider's constructor.

**Base class:** `InferenceModel` (ABC) in `LLMFactory/providers/base.py` defines the contract. Subclasses implement:
- `_load_model()` — initialize SDK client (called automatically in `__init__`)
- `_get_provider()` — return provider name string
- `invoke()` — generate text/embeddings

**Provider modules** live in `LLMFactory/providers/`. Each file contains one or two provider classes. Factory keys map to classes:

| Key | Class | Module |
|-----|-------|--------|
| `ollama` | `OllamaInference` | `providers/ollama.py` |
| `ollama-embed` | `OllamaEmbedInference` | `providers/ollama.py` |
| `lmstudio` | `LMStudioInference` | `providers/lmstudio.py` |
| `anthropic` | `AnthropicInference` | `providers/anthropic.py` |
| `anthropic-bedrock` | `AnthropicBedrockInference` | `providers/anthropic.py` |
| `openai` | `OpenAIInference` | `providers/openai.py` |
| `custom_oai` | `CustomOAIInference` | `providers/openai.py` |
| `gemini` | `GeminiInference` | `providers/gemini.py` |
| `sentence-transformer` | `SentenceTransformerInference` | `providers/embeddings.py` |
| `llamacpp` | `LlamacppInference` | `providers/llamacpp.py` |

**Key patterns:**
- **Lazy imports:** All SDK imports are deferred inside `_load_model()`/`invoke()` to avoid import errors for uninstalled SDKs.
- **Streaming:** `invoke(streaming=True)` returns an `Iterator[str]` generator. Each provider uses a nested `_gen()` closure pattern.
- **Thinking mode:** `use_thinking` (bool or `"low"/"medium"/"high"`) and `return_thinking` params produce a `ThinkingResponse` dataclass with `.content` and `.thinking` fields.
- **Schema/structured output:** `schema: Optional[BaseModel]` uses `model_json_schema()` to pass JSON schema to provider APIs.
- **Images:** `images: Optional[List[Union[str, bytes]]]` are base64-encoded via `_encode_image()` from `base.py`.
- **LMStudio singleton:** Thread-safe instance caching (`_lmstudio_cache` in `llm.py`) handles the SDK's singleton constraint.

## Adding a New Provider

1. Create `LLMFactory/providers/your_provider.py` — subclass `InferenceModel`
2. Export in `LLMFactory/providers/__init__.py`
3. Register key in `LLMModelFactory._models` dict in `LLMFactory/llm.py`
4. Add `tests/test_your_provider.py`

## Testing Conventions

- All tests use `unittest.mock` (`Mock`, `MagicMock`, `monkeypatch`)
- `tests/conftest.py` provides shared fixtures: mock env vars, sample messages, sample images, mock clients
- Tests mock SDK clients via `monkeypatch.setattr` or `sys.modules` injection
- Test files follow `test_<provider>.py` naming

## Code Style

- Line length: 120 (black configured)
- Apache 2.0 license header on every source file
- Full type annotations using `typing` module on all public method signatures
