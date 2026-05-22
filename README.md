# Copilota

Local code assistant with RAG, multi-language support, and vector search.

## Architecture

```
src/copilota/
├── cli.py              # CLI (Click)
├── config.py           # YAML configuration loading
├── core/
│   ├── embedder.py     # Embedding generator (sentence-transformers or mock)
│   ├── indexer.py      # Orchestrates: git repo → parser → chunks → vector DB
│   ├── rag.py          # RAG pipeline: retriever + LLM → response
│   └── retriever.py    # Vector search with filters
├── llm/
│   ├── base.py         # LLM interface (abstract)
│   ├── factory.py      # Factory: creates the correct LLM based on config
│   ├── ollama.py       # Stub LLM (test mode, no server)
│   └── ollama_real.py  # Real Ollama (HTTP to local API)
├── parser/
│   ├── base.py         # BaseParser interface
│   ├── registry.py     # ParserRegistry (dynamic registration)
│   ├── python.py       # Python parser
│   ├── javascript.py   # JavaScript/TypeScript parser
│   ├── php.py          # PHP parser
│   ├── go.py           # Go parser
│   └── rust.py         # Rust parser
└── storage/
    ├── models.py       # ASTNode, CodeChunk, NodeType
    └── vector_db.py    # ChromaDB wrapper
config/
└── default.yaml        # Default configuration
```

## Installation

### Local (from clone)

```bash
git clone https://github.com/HernoDev/copilota.git
cd copilota
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install -e ".[dev]"  # for development
```

### Directly from GitHub (without cloning)

```bash
pip install git+https://github.com/HernoDev/copilota.git
```

With development dependencies:

```bash
pip install "git+https://github.com/HernoDev/copilota.git[dev]"
```

## Migrating to another machine (VM)

To copy the project without the venv or cache files:

```bash
tar czf copilota.tar.gz copilota/ \
  --exclude=copilota/venv \
  --exclude=copilota/__pycache__ \
  --exclude=copilota/.pytest_cache \
  --exclude=copilota/src/copilota.egg-info

# On the destination machine:
tar xzf copilota.tar.gz
cd copilota
python3 -m venv venv && source venv/bin/activate
pip install -e .
```

## Usage

### Indexing a repository

```bash
copilota index /path/to/repo --mock-embeddings
```

### Searching for relevant code

```bash
copilota search "how auth works" --mock-embeddings
copilota search "database connection" -l python -k 10 --mock-embeddings
```

### Asking questions with RAG

```bash
copilota ask "How does the login system work?" --mock-embeddings
```

### Viewing information

```bash
copilota info --mock-embeddings
```

> `--mock-embeddings` uses hash vectors instead of sentence-transformers. Useful for testing without downloading models.

## Configuration

The `config/default.yaml` file controls LLM behavior:

```yaml
llm:
  enabled: false          # false = test mode (stub), true = real LLM
  provider: ollama        # provider: "ollama" (extensible)
  model: qwen2.5-coder    # model to use
  base_url: http://localhost
  port: 11434
  api_path: /api/generate
  chat_api_path: /api/chat
  temperature: 0.7
  max_tokens: 2048
  timeout: 120
```

### Test mode (default)

With `enabled: false` (or without config file), Copilota uses a stub that simulates responses. No server is needed. Ideal for development and CI.

### Real mode (Ollama)

1. Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
2. Download model: `ollama pull qwen2.5-coder`
3. Create `my_config.yaml`:

```yaml
llm:
  enabled: true
  provider: ollama
  model: qwen2.5-coder
```

4. Use with CLI:

```bash
copilota ask "How does auth work?" -c my_config.yaml
copilota info -c my_config.yaml
```

All commands accept `-c / --config` to point to a custom YAML file.

## How to add a new language

1. Create `src/copilota/parser/mylang.py`:

```python
from pathlib import Path

import tree_sitter_mylang as tsm
from tree_sitter import Language, Parser
from copilota.parser.base import BaseParser
from copilota.parser.registry import ParserRegistry
from copilota.storage.models import ASTNode, NodeType

TS_LANGUAGE = Language(tsm.language())

@ParserRegistry.register
class MyLangParser(BaseParser):
    @property
    def language(self) -> str:
        return "mylang"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".ml",)

    def parse_file(self, filepath: Path, source: str) -> list[ASTNode]:
        ts_parser = Parser(TS_LANGUAGE)
        tree = ts_parser.parse(source.encode())
        nodes: list[ASTNode] = []
        self._walk(tree.root_node, source, str(filepath), nodes)
        return nodes

    def _walk(self, node, source, filepath, out):
        ast_node = self._to_ast_node(node, source, filepath)
        if ast_node:
            out.append(ast_node)
        for child in node.children:
            self._walk(child, source, filepath, out)

    def _to_ast_node(self, node, source, filepath) -> ASTNode | None:
        mapping = {
            "function_definition": NodeType.FUNCTION,
            "class_definition": NodeType.CLASS,
        }
        node_type = mapping.get(node.type)
        if not node_type:
            return None
        name = self._extract_name(node)
        return ASTNode(
            node_type=node_type, name=name or "<anonymous>",
            source_code=node.text.decode(),
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            filepath=filepath, language=self.language,
        )

    def _extract_name(self, node) -> str | None:
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode()
        return None

    def get_chunk_text(self, node: ASTNode) -> str:
        return node.source_code
```

2. Install the tree-sitter grammar:

```bash
pip install tree-sitter-mylang
```

3. Import in the CLI (`src/copilota/cli.py`), add to `_import_parsers()`:

```python
def _import_parsers():
    from copilota.parser import python, javascript, php, go, rust, mylang
```

## How to add a new LLM provider

The system is extensible for any HTTP API provider.

1. Create `src/copilota/llm/myprovider.py`:

```python
from copilota.config import LLMConfig
from copilota.llm.base import BaseLLM


class MyProviderLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        self.config = config

    async def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None):
        import httpx
        # Your HTTP logic here
        async with httpx.AsyncClient() as client:
            resp = await client.post(self.config.generate_url, json={...})
            return resp.json()["text"]

    async def chat(self, messages, temperature=None, max_tokens=None):
        import httpx
        # Your HTTP logic here
        async with httpx.AsyncClient() as client:
            resp = await client.post(self.config.chat_url, json={...})
            return resp.json()["message"]["content"]
```

2. Register in the factory (`src/copilota/llm/factory.py`):

```python
def create_llm(config: AppConfig) -> BaseLLM:
    if not config.llm.enabled:
        return OllamaStub()
    if config.llm.provider == "ollama":
        return OllamaLLM(config.llm)
    if config.llm.provider == "myprovider":
        from copilota.llm.myprovider import MyProviderLLM
        return MyProviderLLM(config.llm)
    raise ValueError(f"Unsupported LLM provider: {config.llm.provider}")
```

3. Use in config:

```yaml
llm:
  enabled: true
  provider: myprovider
  model: my-model
  base_url: https://api.myprovider.com
  port: 443
```

## Stack

| Layer | Technology |
|-------|------------|
| Parsing | tree-sitter (30+ languages) |
| Vector DB | ChromaDB (local, persistent) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Ollama (configurable, extensible) |
| CLI | Click + Rich |
| Config | YAML (pyyaml) |
| API | FastAPI (pending) |

## Tests

```bash
python -m pytest tests/ -v
```

26 tests, all passing.
