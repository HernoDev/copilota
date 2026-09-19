# Job: Correcciones Copilota v0.2 → v0.3

## Tareas (en orden de ejecución)

### T1: Fix config paths + precedence
- **Problema**: Fallback al package config apunta a ruta inexistente (`src/copilota/config/default.yaml`). En instalación standalone, `inspect.getfile` cae en site-packages sin `config/`. Además, un `config/default.yaml` en cwd sombra silenciosamente a `~/.copilota/config.yaml`.
- **Fix**: 
  - Usar `importlib.resources` para localizar el default empaquetado (o eliminar el fallback si no se empaqueta).
  - Invertir precedencia: explicit > global (~/.copilota) > project-local > package.
  - Loguear qué archivo ganó.
  - Validar tipos (port int, temperature float) con mensaje amigable.

### T2: Deps muertas + sentence-transformers optional
- **Problema**: `fastapi`, `uvicorn`, `pydantic` declaradas pero nunca importadas. `sentence-transformers>=2.2` como hard dep descarga ~2GB de torch aunque uses mock mode.
- **Fix**:
  - Quitar fastapi/uvicorn/pydantic de dependencies.
  - Mover sentence-transformers a `[project.optional-dependencies] embed = [...]`.
  - El código ya maneja ImportError gracefully (`_check_mock_needed`).

### T3: Incluir IMPORT/MODULE/VARIABLE en chunking
- **Problema**: `CHUNKED_NODE_TYPES` excluye IMPORT, MODULE, VARIABLE, CONSTANT. Preguntas tipo "¿cómo está configurado X?" no tienen chunks.
- **Fix**: Agregar NodeType.IMPORT, NodeType.MODULE, NodeType.VARIABLE a CHUNKED_NODE_TYPES. Ajustar `get_chunk_text` en parsers que no los manejan.

### T4: Tests → tmp_path (no mutar index real)
- **Problema**: Fixtures de test_cli.py y test_storage.py usan `VectorStore()` real (~/.local/share/copilota). Correr la suite borra y reseedea el index de producción.
- **Fix**: Usar `tmp_path` fixture de pytest para crear VectorStore temporal en cada test.

### T5: Tests HTTP con httpx.MockTransport
- **Problema**: Cero tests para openai_compat.py y ollama_real.py. La parsing de respuestas asume estructuras fijas que pueden variar entre servidores.
- **Fix**: Agregar tests con MockTransport que simulen respuestas exitosas, errores, content=null, etc.

### T6: LLM clients robustos
- **Problema**: `openai_compat.py` asume `choices[0].message.content` sin fallback. `ollama_real.py` mismo. `_fetch_models` usa `/v1/models` que no existe en Ollama nativo (debería ser `/api/tags`).
- **Fix**:
  - Parsing defensivo con `.get()` y mensajes de error contextuales.
  - `_fetch_models`: dispatch por provider (ollama → `/api/tags`, openai → `/v1/models`).
  - Retry simple en connection refused para ollama_real.

### T7: Parser fixes (rust impl, go type_decl, TS grammar)
- **Problema**: Rust `impl_item → CLASS` es mislabel + name extraction falla. Go `type_declaration → STRUCT` incluye interfaces. JS parser usa grammar JavaScript para .ts/.tsx.
- **Fix**:
  - Rust: `impl_item` → nuevo tipo o CLASS con mejor name extraction (buscar `trait_ref`/`type_path`).
  - Go: separar `type_declaration` en struct vs interface según contenido.
  - JS: agregar `tree-sitter-typescript` como dep y usar grammar correcta para .ts/.tsx.

### T8: Performance — cache Parser + batch add + list_repos
- **Problema**: Cada `parse_file` crea `Parser(TS_LANGUAGE)` nuevo. `add_chunks` hace un add por archivo. `list_repos()` carga toda la colección en RAM.
- **Fix**:
  - Cache del objeto Parser por instancia de parser (thread-safe en tree-sitter).
  - Batching en indexer: acumular chunks y hacer adds en lotes de 512.
  - `list_repos()`: usar `get(include=["metadatas"])` en vez de `get()`.

### T9: CLI improvements
- **Problema**: `--mock-embeddings` repetido en 4 commands. Version hardcoded. Sin `copilota delete <repo>`. `asyncio.run` break en async context.
- **Fix**:
  - Mover `--mock-embeddings` a group option.
  - Leer version de `importlib.metadata`.
  - Agregar comando `delete <repo>`.
  - Reemplazar `asyncio.run` por wrapper seguro (anyio.from_thread o event loop detection).

### T10: RAG pipeline improvements
- **Problema**: Sin threshold de relevancia. Sin budget de tokens. Context assembly naive (puede explotar ventana de modelo local).
- **Fix**:
  - Agregar `min_score` threshold (default 0.3) — filtrar resultados bajo el umbral.
  - Agregar `max_context_chars` (default ~8000) — truncar contexto si excede.
  - Configurable vía config.yaml.

---

## Proceso por tarea

Para cada tarea:
1. Implementar el fix en `src/copilota/`
2. Ajustar/agregar tests
3. Ejecutar `pytest` + `ruff check`
4. Ejecutar `./install.sh` para actualizar la instalación
5. Verificar desde la instalación (`~/.local/copilota/bin/copilota`)
6. Registrar en `informe_correciones.md`: problema existente + resolución aplicada
