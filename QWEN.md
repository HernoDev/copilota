# Copilota - Contexto del Proyecto

## Resumen del Proyecto

**Copilota** es un asistente de codigo local que utiliza Retrieval Augmented Generation (RAG) para responder preguntas sobre codigo indexado. Soporta multiples lenguajes de programacion (Python, JavaScript/TypeScript, PHP, Go, Rust) y permite:

- Indexar repositorios Git
- Buscar fragmentos de codigo relevantes mediante busqueda vectorial
- Responder preguntas sobre el codigo usando un LLM local (Ollama)

### Tecnologias Clave

| Capa | Tecnologia |
|------|------------|
| Parsing | tree-sitter (30+ lenguajes) |
| Vector DB | ChromaDB (local, persistente) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) o mock |
| LLM | Ollama (configurable, extensible) |
| CLI | Click + Rich |
| Config | YAML (pyyaml) |

### Arquitectura

```
src/copilota/
├── cli.py              # CLI (Click)
├── config.py           # Carga de configuracion YAML
├── core/
│   ├── embedder.py     # Generador de embeddings
│   ├── indexer.py      # Orquesta: git repo → parser → chunks → vector DB
│   ├── rag.py          # Pipeline RAG: retriever + LLM → respuesta
│   └── retriever.py    # Busqueda vectorial con filtros
├── llm/
│   ├── base.py         # Interfaz LLM (abstracta)
│   ├── factory.py      # Factory: crea el LLM correcto segun config
│   ├── ollama.py       # Stub LLM (modo test, sin servidor)
│   └── ollama_real.py  # Ollama real (HTTP a API local)
├── parser/
│   ├── base.py         # Interfaz BaseParser
│   ├── registry.py     # ParserRegistry (registro dinamico)
│   ├── python.py       # Parser Python
│   ├── javascript.py   # Parser JS/TS
│   ├── php.py          # Parser PHP
│   ├── go.py           # Parser Go
│   └── rust.py         # Parser Rust
└── storage/
    ├── models.py       # ASTNode, CodeChunk, NodeType
    └── vector_db.py    # Wrapper ChromaDB
config/
└── default.yaml        # Configuracion por defecto
```

---

## Construccion y Ejecucion

### Instalacion

```bash
# Clonar y activar entorno virtual
git clone https://github.com/HernoDev/copilota.git
cd copilota
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -e .
pip install -e ".[dev]"  # dependencias de desarrollo
```

### Comandos de la CLI

```bash
# Indexar un repositorio
copilota index /path/to/repo --mock-embeddings

# Buscar codigo relevante
copilota search "como funciona auth" --mock-embeddings
copilota search "database connection" -l python -k 10 --mock-embeddings

# Preguntar con RAG
copilota ask "Como funciona el sistema de login?" --mock-embeddings

# Ver informacion del indice
copilota info --mock-embeddings
```

### Ejecucion de Tests

```bash
python -m pytest tests/ -v
```

---

## Convenciones de Desarrollo

### Estilo de Codigo

- **Formateo**: No hay configuracion de black o isort explicita; se usa ruff para linting.
- **Longitud de linea**: 100 caracteres (definido en pyproject.toml).
- **Linting**: ruff con reglas E (pycodestyle), F (pyflakes), I (isort).

### Patrones de Diseno

1. **Registro Dinamico de Parsers**: Los parsers se registran mediante decorador @ParserRegistry.register para descubrimiento automatico.

2. **Factory Pattern**: La clase create_llm() en llm/factory.py crea instancias de LLM segun la configuracion.

3. **Strategy Pattern**: Los parsers implementan BaseParser con metodos abstractos parse_file() y get_chunk_text().

4. **Mock/Real Dual Mode**: El sistema soporta tanto embeddings mock (para testing) como reales (sentence-transformers).

### Pruebas

- Los tests estan en tests/ y usan pytest.
- Se usa pytest-asyncio para tests asincronos.
- Los tests deben pasar antes de cualquier cambio significativo.

### Configuracion

- El archivo config/default.yaml controla la configuracion del LLM.
- Por defecto, llm.enabled = false usa un stub de prueba.
- Para usar LLM real, establecer llm.enabled = true y configurar provider, model, base_url.

---

## Extensibilidad

### Agregar un Nuevo Lenguaje

1. Crear un parser en src/copilota/parser/mi_lenguaje.py que implemente BaseParser.
2. Instalar el grammar de tree-sitter: pip install tree-sitter-mylang.
3. Importar en src/copilota/cli.py dentro de _import_parsers().

### Agregar un Nuevo Proveedor LLM

1. Crear src/copilota/llm/mi_proveedor.py que implemente BaseLLM.
2. Registrar en src/copilota/llm/factory.py dentro de create_llm().
3. Configurar en config/default.yaml con provider: mi_proveedor.

---

## Notas Importantes

### Modo Mock vs Modo Real

- **Mock** (--mock-embeddings): Usa vectores hash en vez de sentence-transformers. Ideal para testing sin descargar modelos.
- **Real**: Requiere sentence-transformers y ollama con el modelo especificado en la config.

### Seguridad

- Nunca exponer credenciales o API keys en el codigo.
- El LLM real usa Ollama local; no hay exposicion a servicios externos si se configura correctamente.

### Git

- El proyecto esta versionado con Git.
- Los cambios deben estar probados antes de hacer commit.
- Revisar git status y git diff antes de cada commit.

---

## Recursos

- **README**: README_es.md (documentacion en español)
- **Configuracion**: config/default.yaml
- **Tests**: tests/
