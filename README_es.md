# Copilota

Asistente de código local con RAG, multi-lenguaje y vector search.

## Arquitectura

```
src/copilota/
├── cli.py              # CLI (Click)
├── config.py           # Carga de configuración YAML
├── core/
│   ├── embedder.py     # Generador de embeddings (sentence-transformers o mock)
│   ├── indexer.py      # Orquesta: git repo → parser → chunks → vector DB
│   ├── rag.py          # Pipeline RAG: retriever + LLM → respuesta
│   └── retriever.py    # Búsqueda vectorial con filtros
├── llm/
│   ├── base.py         # Interfaz LLM (abstracta)
│   ├── factory.py      # Factory: crea el LLM correcto según config
│   ├── ollama.py       # Stub LLM (modo test, sin servidor)
│   └── ollama_real.py  # Ollama real (HTTP a API local)
├── parser/
│   ├── base.py         # Interfaz BaseParser
│   ├── registry.py     # ParserRegistry (registro dinámico)
│   ├── python.py       # Parser Python
│   ├── javascript.py   # Parser JS/TS
│   ├── php.py          # Parser PHP
│   ├── go.py           # Parser Go
│   └── rust.py         # Parser Rust
└── storage/
    ├── models.py       # ASTNode, CodeChunk, NodeType
    └── vector_db.py    # Wrapper ChromaDB
config/
└── default.yaml        # Configuración por defecto
```

## Instalación

### Local (desde clon)

```bash
git clone https://github.com/HernoDev/copilota.git
cd copilota
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install -e ".[dev]"  # para desarrollo
```

### Directamente desde GitHub (sin clonar)

```bash
pip install git+https://github.com/HernoDev/copilota.git
```

Con dependencias de desarrollo:

```bash
pip install "git+https://github.com/HernoDev/copilota.git[dev]"
```

## Trasladar a otra máquina (VM)

Para copiar el proyecto sin el venv ni archivos de cache:

```bash
tar czf copilota.tar.gz copilota/ \
  --exclude=copilota/venv \
  --exclude=copilota/__pycache__ \
  --exclude=copilota/.pytest_cache \
  --exclude=copilota/src/copilota.egg-info

# En la máquina destino:
tar xzf copilota.tar.gz
cd copilota
python3 -m venv venv && source venv/bin/activate
pip install -e .
```

## Uso

### Indexar un repositorio

```bash
copilota index /path/to/repo --mock-embeddings
```

### Buscar código relevante

```bash
copilota search "cómo funciona auth" --mock-embeddings
copilota search "database connection" -l python -k 10 --mock-embeddings
```

### Preguntar con RAG

```bash
copilota ask "¿Cómo funciona el sistema de login?" --mock-embeddings
```

### Ver información

```bash
copilota info --mock-embeddings
```

> `--mock-embeddings` usa vectores hash en vez de sentence-transformers. Útil para testing sin descargar modelos.

## Configuración YAML

El archivo `config/default.yaml` controla el comportamiento del LLM:

```yaml
llm:
  enabled: false          # false = modo test (stub), true = LLM real
  provider: ollama        # proveedor: "ollama" (extensible)
  model: qwen2.5-coder    # modelo a usar
  base_url: http://localhost
  port: 11434
  api_path: /api/generate
  chat_api_path: /api/chat
  temperature: 0.7
  max_tokens: 2048
  timeout: 120
```

### Modo test (por defecto)

Con `enabled: false` (o sin archivo de config), Copilota usa un stub que simula respuestas. No necesita ningún servidor corriendo. Ideal para desarrollo y CI.

### Modo real (Ollama)

1. Instalar Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
2. Descargar modelo: `ollama pull qwen2.5-coder`
3. Crear `mi_config.yaml`:

```yaml
llm:
  enabled: true
  provider: ollama
  model: qwen2.5-coder
```

4. Usar con la CLI:

```bash
copilota ask "¿Cómo funciona el auth?" -c mi_config.yaml
copilota info -c mi_config.yaml
```

Todos los comandos aceptan `-c / --config` para apuntar a un archivo YAML personalizado.

## Cómo agregar un nuevo lenguaje

1. Crear `src/copilota/parser/mi_lenguaje.py`:

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

2. Instalar el grammar de tree-sitter:

```bash
pip install tree-sitter-mylang
```

3. Importar en la CLI (`src/copilota/cli.py`), agregar al `_import_parsers()`:

```python
def _import_parsers():
    from copilota.parser import python, javascript, php, go, rust, mi_lenguaje
```

## Cómo agregar un proveedor LLM nuevo

El sistema es extensible para cualquier proveedor que tenga una API HTTP.

1. Crear `src/copilota/llm/mi_proveedor.py`:

```python
from copilota.config import LLMConfig
from copilota.llm.base import BaseLLM


class MiProveedorLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        self.config = config

    async def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None):
        import httpx
        # Tu lógica HTTP aquí
        async with httpx.AsyncClient() as client:
            resp = await client.post(self.config.generate_url, json={...})
            return resp.json()["text"]

    async def chat(self, messages, temperature=None, max_tokens=None):
        import httpx
        # Tu lógica HTTP aquí
        async with httpx.AsyncClient() as client:
            resp = await client.post(self.config.chat_url, json={...})
            return resp.json()["message"]["content"]
```

2. Registrar en la factory (`src/copilota/llm/factory.py`):

```python
def create_llm(config: AppConfig) -> BaseLLM:
    if not config.llm.enabled:
        return OllamaStub()
    if config.llm.provider == "ollama":
        return OllamaLLM(config.llm)
    if config.llm.provider == "mi_proveedor":
        from copilota.llm.mi_proveedor import MiProveedorLLM
        return MiProveedorLLM(config.llm)
    raise ValueError(f"Proveedor LLM no soportado: {config.llm.provider}")
```

3. Usar en config:

```yaml
llm:
  enabled: true
  provider: mi_proveedor
  model: mi-modelo
  base_url: https://api.mi-proveedor.com
  port: 443
```

## Stack

| Capa | Tecnología |
|------|-----------|
| Parsing | tree-sitter (30+ lenguajes) |
| Vector DB | ChromaDB (local, persistente) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Ollama (configurable, extensible) |
| CLI | Click + Rich |
| Config | YAML (pyyaml) |
| API | FastAPI (pendiente) |

## Tests

```bash
python -m pytest tests/ -v
```

26 tests, todos pasando.
