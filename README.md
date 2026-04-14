# Copilota

Asistente de código local con RAG, multi-lenguaje y vector search.

## Arquitectura

```
src/copilota/
├── cli.py              # CLI (Click)
├── core/
│   ├── embedder.py     # Generador de embeddings (sentence-transformers o mock)
│   ├── indexer.py      # Orquesta: git repo → parser → chunks → vector DB
│   ├── rag.py          # Pipeline RAG: retriever + LLM → respuesta
│   └── retriever.py    # Búsqueda vectorial con filtros
├── llm/
│   ├── base.py         # Interfaz LLM (abstracta)
│   └── ollama.py       # Stub Ollama (implementación futura)
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
```

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install -e ".[dev]"  # para desarrollo
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

## Cómo agregar un nuevo lenguaje

1. Crear `src/copilota/parser/mi_lenguaje.py`:

```python
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

## Agregar LLM real (Ollama)

El stub en `src/copilota/llm/ollama.py` simula respuestas. Para conectar Ollama real:

1. Instalar Ollama en la VM
2. Descargar modelo: `ollama pull qwen2.5-coder`
3. Reemplazar `OllamaLLM.generate()` y `chat()` con llamadas HTTP a `http://localhost:11434/api/generate`

## Stack

| Capa | Tecnología |
|------|-----------|
| Parsing | tree-sitter (30+ lenguajes) |
| Vector DB | ChromaDB (local, persistente) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Ollama (stub, conectar después) |
| CLI | Click + Rich |
| API | FastAPI (pendiente) |

## Tests

```bash
python -m pytest tests/ -v
```

26 tests, todos pasando.
