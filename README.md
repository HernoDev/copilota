# Copilota

Local code assistant with RAG, multi-language support, and vector search.

## Features

- ✅ **Multi-language parsing**: Python, JavaScript, PHP, Go, Rust (extensible)
- ✅ **Local RAG**: Ask questions about your codebase
- ✅ **Vector search**: Find relevant code snippets
- ✅ **Persistent storage**: Code index saved in `~/.local/share/copilota`
- ✅ **Ollama integration**: Local LLM with configurable models
- ✅ **Automatic config loading**: No need for `-c` flag
- ✅ **Extensible architecture**: Add new languages or LLM providers

## Architecture

```
src/copilota/
├── cli.py              # CLI (Click)
├── config.py           # YAML configuration loader
├── core/
│   ├── embedder.py     # Embedding generator (sentence-transformers or mock)
│   ├── indexer.py      # Orchestrates: git repo → parser → chunks → vector DB
│   ├── rag.py          # RAG pipeline: retriever + LLM → answer
│   └── retriever.py    # Vector search with filters
├── llm/
│   ├── base.py         # LLM interface (abstract)
│   ├── factory.py      # Factory: creates correct LLM based on config
│   ├── ollama.py       # Stub LLM (test mode, no server)
│   └── ollama_real.py  # Real Ollama (HTTP to local API)
├── parser/
│   ├── base.py         # BaseParser interface
│   ├── registry.py     # ParserRegistry (dynamic registration)
│   ├── python.py       # Parser Python
│   ├── javascript.py   # Parser JS/TS
│   ├── php.py          # Parser PHP
│   ├── go.py           # Parser Go
│   └── rust.py         # Parser Rust
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

### Directly from GitHub (no clone)

```bash
pip install git+https://github.com/HernoDev/copilota.git
```

With development dependencies:

```bash
pip install "git+https://github.com/HernoDev/copilota.git[dev]"
```

### Standalone install (taskrunner / RAG)

`install.sh` creates a dedicated venv in `~/.local/copilota`, installs the
package (non-editable), verifies the binary, and configures the taskrunner's
RAG settings if it is installed. Idempotent — re-run it to update the
installation after code changes:

```bash
./install.sh                        # installs to ~/.local/copilota
COPILOTA_HOME=/other/prefix ./install.sh
```

## Move to Another Machine (VM)

To copy the project without venv or cache files:

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

### Index a Repository

```bash
copilota index /path/to/repo
```

### Search for Relevant Code

```bash
copilota search "how auth works"
copilota search "database connection" -l python -k 10
```

### Ask Questions with RAG

```bash
copilota ask "How does the login system work?"
```

### View Information

```bash
copilota info
```

## Ollama Integration

### Setup

1. **Install Ollama**:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Pull a model**:
   ```bash
   ollama pull qwen35:27b_64k
   ```

3. **Create global configuration** (optional, for automatic loading):
   ```bash
   mkdir -p ~/.copilota
   cat > ~/.copilota/config.yaml << 'EOF'
   llm:
     enabled: true
     provider: ollama
     model: qwen35:27b_64k
     base_url: http://localhost
     port: 11434
     temperature: 0.7
     max_tokens: 2048
     timeout: 120
   EOF
   ```

4. **Use with CLI** (no `-c` flag needed if config exists):
   ```bash
   copilota ask "How does auth work?"
   ```

### Configuration Loading Order

Copilota automatically loads configuration from the first existing file in this order:

1. File specified with `-c / --config` flag
2. `config/default.yaml` in current directory
3. `~/.copilota/config.yaml` (global user config)
4. Package installation directory

This means you can set up a global configuration once and use Copilota anywhere without flags.

## Configuration

The YAML configuration file controls LLM behavior:

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

### Test Mode (Default)

With `enabled: false` (or no config file), Copilota uses a stub that simulates responses. No server needed. Ideal for development and CI.

### Real Mode (Ollama)

Once you have Ollama running and a global config set up, simply run:

```bash
copilota ask "How does the auth work?"
```

No `-c` flag needed! The system will automatically find your configuration.

## How to Add a New Language

1. Create `src/copilota/parser/mi_lenguaje.py`:

```python
from pathlib import Path

import tree_sitter_mylang as tsm
from tree_sitter import Language, Parser
from copilota.parser.base import BaseParser
from copilota.parser.registry import ParserRegistry
from copilota.storage.models import ASTNode, NodeType

# Define the tree-sitter language
TS_LANGUAGE = Language(tsm.language())

# Register the parser
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
        # Map tree-sitter node types to our AST node types
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
    from copilota.parser import python, javascript, php, go, rust, mi_lenguaje
```

## How to Add a New LLM Provider

The system is extensible for any provider with an HTTP API.

1. Create `src/copilota/llm/mi_proveedor.py`:

```python
from copilota.config import LLMConfig
from copilota.llm.base import BaseLLM


class MiProveedorLLM(BaseLLM):
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
    if config.llm.provider == "mi_proveedor":
        from copilota.llm.mi_proveedor import MiProveedorLLM
        return MiProveedorLLM(config.llm)
    raise ValueError(f"Unsupported LLM provider: {config.llm.provider}")
```

3. Use in config:

```yaml
llm:
  enabled: true
  provider: mi_proveedor
  model: mi-modelo
  base_url: https://api.mi-proveedor.com
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
