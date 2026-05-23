# Copilota - Local Code Assistant with RAG

## Project Overview

**Copilota** is a local code assistant that uses Retrieval-Augmented Generation (RAG) to answer questions about codebases. It supports multiple programming languages through tree-sitter parsers and stores code embeddings in a local ChromaDB vector database.

### Key Features

- **Multi-language support**: Python, JavaScript/TypeScript, PHP, Go, Rust (extensible to 30+ languages via tree-sitter)
- **Local-first**: All processing happens locally - no cloud dependencies
- **Git-aware**: Indexes tracked files from Git repositories
- **Dual LLM mode**: Mock mode for testing (no server needed) or real Ollama integration
- **Extensible architecture**: Easy to add new languages or LLM providers

### Architecture

```
src/copilota/
├── cli.py              # Click-based CLI interface
├── config.py           # YAML configuration loading
├── core/
│   ├── embedder.py     # Embedding generation (sentence-transformers or mock)
│   ├── indexer.py      # Orchestrates: git repo → parser → chunks → vector DB
│   ├── rag.py          # RAG pipeline: retriever + LLM → answer
│   └── retriever.py    # Vector search with filters
├── llm/
│   ├── base.py         # Abstract LLM interface
│   ├── factory.py      # Factory pattern for LLM creation
│   ├── ollama.py       # Stub LLM (test mode, no server)
│   └── ollama_real.py  # Real Ollama HTTP client
├── parser/
│   ├── base.py         # BaseParser abstract class
│   ├── registry.py     # ParserRegistry for dynamic registration
│   ├── python.py       # Python parser
│   ├── javascript.py   # JavaScript/TypeScript parser
│   ├── php.py          # PHP parser
│   ├── go.py           # Go parser
│   └── rust.py         # Rust parser
└── storage/
    ├── models.py       # ASTNode, CodeChunk, NodeType dataclasses
    └── vector_db.py    # ChromaDB wrapper
```

## Building and Running

### Installation

```bash
# Clone and install locally
git clone https://github.com/HernoDev/copilota.git
cd copilota
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install -e ".[dev]"  # for development with tests/linters

# Or install directly from GitHub
pip install git+https://github.com/HernoDev/copilota.git
```

### Commands

```bash
# Index a repository
copilota index /path/to/repo --mock-embeddings

# Search for relevant code
copilota search "how auth works" --mock-embeddings
copilota search "database connection" -l python -k 10 --mock-embeddings

# Ask questions (RAG)
copilota ask "How does the login system work?" --mock-embeddings

# Show info
copilota info --mock-embeddings
```

### Configuration

The `config/default.yaml` controls LLM behavior:

```yaml
llm:
  enabled: false          # false = mock mode, true = real LLM
  provider: ollama
  model: qwen2.5-coder
  base_url: http://localhost
  port: 11434
  temperature: 0.7
  max_tokens: 2048
```

**Mock mode** (default): Uses hash-based embeddings and stub LLM responses. No external services needed.

**Real mode**: Requires Ollama running locally with the specified model.

### Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src/copilota

# Lint with ruff
ruff check src/
```

## Development Conventions

### Code Style

- **Formatter**: Black-compatible (line length 100)
- **Linter**: Ruff with rules E, F, I
- **Type hints**: Required for all new code
- **Docstrings**: Use triple-quoted strings at module/class/function level

### Testing Practices

- **Unit tests**: Located in `tests/` directory
- **Async support**: Uses `pytest-asyncio` with `asyncio_mode = "auto"`
- **Mock-first**: Tests use mock embeddings by default (no model downloads)
- **Coverage**: Aim for high coverage on core logic

### Adding New Languages

1. Create `src/copilota/parser/newlang.py` extending `BaseParser`
2. Install tree-sitter grammar: `pip install tree-sitter-newlang`
3. Import in `cli.py` `_import_parsers()` function
4. Register via `@ParserRegistry.register` decorator

### Adding New LLM Providers

1. Create `src/copilota/llm/newprovider.py` implementing `BaseLLM`
2. Register in `llm/factory.py` `create_llm()` function
3. Add provider name to config options

### Key Design Patterns

- **Factory Pattern**: LLM creation via `create_llm()` factory function
- **Registry Pattern**: Parsers registered dynamically via `ParserRegistry`
- **Strategy Pattern**: Embedding generation (mock vs real)
- **Repository Pattern**: VectorStore abstracts ChromaDB operations

## Tech Stack

| Layer | Technology |
|-------|-------|
| Parsing | tree-sitter (language grammars) |
| Vector DB | ChromaDB (local, persistent) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Ollama (configurable, extensible) |
| CLI | Click + Rich |
| Config | YAML (pyyaml) |
| Tests | pytest, pytest-asyncio |
| Linting | Ruff |

## Important Files

- `src/copilota/cli.py`: Main CLI entry point with all commands
- `src/copilota/config.py`: Configuration loading and validation
- `src/copilota/core/rag.py`: RAG pipeline orchestration
- `src/copilota/core/indexer.py`: Repository indexing logic
- `src/copilota/parser/registry.py`: Parser registration and lookup
- `src/copilota/llm/factory.py`: LLM provider factory
- `config/default.yaml`: Default configuration values

## Current Status

- **Version**: 0.2.0
- **Test suite**: 26 tests passing
- **Supported languages**: Python, JavaScript, TypeScript, PHP, Go, Rust
- **LLM providers**: Ollama (mock + real modes)
