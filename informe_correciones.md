# Informe de Correcciones — Copilota v0.2 → v0.3

Registro de problemas encontrados y resueltos durante la sesión de correcciones (2026-09-19).

---

## T1: Fix config paths + precedence

**Problema existente:**
- El fallback al package config usaba `inspect.getfile(__import__("copilota"))` que apuntaba a `src/copilota/config/default.yaml` (ruta inexistente). En la instalación standalone (`~/.local/copilota`, non-editable), `inspect.getfile` cae en site-packages donde no hay directorio `config/`.
- La precedencia era: explicit > project-local > global > package. Esto significaba que un `config/default.yaml` en el cwd (como el del propio repo copilota) **sombra silenciosamente** a `~/.copilota/config.yaml`.
- No había validación de tipos: un `port: "abc"` en YAML crasheaba con traceback crudo.
- No se informaba al usuario qué archivo de config ganó.

**Resolución aplicada:**
- Reemplazado `inspect.getfile` por `importlib.resources.files("copilota")` para localizar el default empaquetado correctamente.
- Invertida la precedencia a: **explicit (-c) > global (~/.copilota) > project (./config) > package > defaults**.
- Agregado campo `config_source` en `AppConfig` que registra de dónde se cargó.
- Extraído `_parse_llm_config()` con validación de tipos y mensaje de error amigable (`ValueError: Config LLM inválida: ...`).
- Agregado `logging` para debug de qué archivo fue seleccionado.

**Archivos modificados:** `src/copilota/config.py`

---

## T2: Deps muertas + sentence-transformers optional

**Problema existente:**
- `fastapi>=0.110`, `uvicorn>=0.29`, `pydantic>=2.6` declaradas en dependencies pero **nunca importadas** en ningún módulo de `src/`. Arrastraban una web stack completa (~50MB) para un CLI tool.
- `sentence-transformers>=2.2` como dependencia hard: descarga ~2GB de torch aunque el usuario solo use mock mode.

**Resolución aplicada:**
- Eliminadas `fastapi`, `uvicorn`, `pydantic` de `dependencies`.
- Movido `sentence-transformers` a `[project.optional-dependencies] embed = ["sentence-transformers>=2.2"]`.
- Agregado `pytest-cov>=5.0` a dev extras (documentado en QWEN.md pero no disponible).
- El código ya manejaba `ImportError` gracefully (`_check_mock_needed` en embedder.py), así que no se rompió nada.

**Archivos modificados:** `pyproject.toml`

---

## T3: Incluir IMPORT/MODULE/VARIABLE en chunking

**Problema existente:**
- `CHUNKED_NODE_TYPES` en `indexer.py` excluía `NodeType.IMPORT`, `NodeType.MODULE`, `NodeType.VARIABLE`, `NodeType.CONSTANT`.
- Los parsers extraían estos nodos pero el indexer los descartaba silenciosamente.
- Preguntas tipo "¿cómo está configurado X?", "¿dónde se define esa constante?", "¿qué importa este módulo?" no tenían chunks asociados.

**Resolución aplicada:**
- Agregados `NodeType.IMPORT`, `NodeType.MODULE`, `NodeType.VARIABLE` a `CHUNKED_NODE_TYPES`.
- Los parsers ya manejan estos tipos con el fallback `return node.source_code` en `get_chunk_text()`, así que no se necesitó modificar parsers.

**Archivos modificados:** `src/copilota/core/indexer.py`

---

## T4: Tests → tmp_path (no mutar index real)

**Problema existente:**
- Los fixtures `store()` en `test_cli.py`, `test_storage.py`, y `test_core.py` creaban `VectorStore()` sin argumento → usaba el path real `~/.local/share/copilota`.
- Correr la suite de tests **borraba y reseedea** el index de producción del usuario.
- Solo `TestIndexer` en `test_core.py` usaba `tmp_store` (con `tmp_path`).

**Resolución aplicada:**
- `test_storage.py`: fixture ahora usa `VectorStore(persist_directory=tmp_path / "chroma")`.
- `test_cli.py`: fixture crea store temporal + monkeypatch de `copilota.cli.VectorStore` para redirigir las llamadas internas de la CLI al store temporal.
- `test_core.py`: fixture `store` ahora usa `tmp_path`.
- Eliminado el pattern `s.clear()` antes/después (innecesario con stores temporales).

**Archivos modificados:** `tests/test_cli.py`, `tests/test_storage.py`, `tests/test_core.py`

---

## T5: Tests HTTP con httpx.MockTransport

**Problema existente:**
- Cero tests para `openai_compat.py` (coverage 46%) y `ollama_real.py` (coverage 29%).
- La parsing de respuestas asumía estructuras fijas (`choices[0]["message"]["content"]`) sin tests que verificaran el comportamiento ante respuestas anómalas.

**Resolución aplicada:**
- Creado `tests/test_llm_clients.py` con 8 tests usando `httpx.MockTransport`:
  - OpenAI: success, content=null, missing choices, HTTP 500.
  - Ollama: generate success, response key missing, chat success, chat content null.
- Los tests inyectan un `httpx.AsyncClient` con MockTransport vía `llm._client` (nuevo atributo agregado en T6).

**Archivos creados:** `tests/test_llm_clients.py`

---

## T6: LLM clients robustos

**Problema existente:**
- `openai_compat.py`: `resp.json()["choices"][0]["message"]["content"]` → `KeyError`/`TypeError` si el servidor devuelve `content: null`, `choices` vacío, o un formato diferente.
- `ollama_real.py`: mismo problema con `resp.json()["response"]` y `["message"]["content"]`.
- `_fetch_models` en `cli.py` usaba `{url}/v1/models` para todos los providers. Para Ollama nativo el endpoint es `/api/tags` → `copilota models` fallaba silenciosamente.
- `httpx` se importaba dentro de métodos en `ollama_real.py` (inconsistente con `openai_compat.py`).

**Resolución aplicada:**
- Parsing defensivo con `.get()` chains: `data.get("choices")` → `choices[0].get("message", {})` → `message.get("content")`. Retorna `""` si falta contenido.
- Agregado atributo `_client` a ambas clases: permite inyectar un client externo (tests) o crear uno interno. Si es externo, no se cierra.
- `httpx` movido a import de módulo en `ollama_real.py`.
- `_fetch_models` ahora dispatch por provider: ollama → `/api/tags` (lee `models[].name`), openai → `/v1/models` (lee `data[].id`).

**Archivos modificados:** `src/copilota/llm/openai_compat.py`, `src/copilota/llm/ollama_real.py`, `src/copilota/cli.py`

---

## T7: Parser fixes (rust impl, go type_decl, TS grammar)

**Problema existente:**
- Rust: `impl_item → CLASS` era mislabel semántico. `_extract_name` buscaba child `name` pero en `impl Foo` el nombre está bajo `type_path`/`trait_ref` → muchos chunks `<anonymous>`.
- Go: `type_declaration → STRUCT` etiquetaba interfaces y type aliases como structs.
- JavaScript: usaba grammar `tree-sitter-javascript` para `.ts`/`.tsx` → nodes ERROR, names/types mal parseados.

**Resolución aplicada:**
- Rust: `_extract_name` ahora busca en `type_path`/`trait_ref` children cuando el nodo es `impl_item`. Nuevo helper `_extract_type_path_name` recursivo para `scoped_identifier`.
- Go: nuevo método `_handle_type_declaration` que inspecciona `type_spec` children: si tiene `interface_type` → `NodeType.INTERFACE`, sino → `NodeType.STRUCT`. Usa el `type_spec` child (no el `type_declaration` wrapper) para source/lines.
- TypeScript: creado parser dedicado `src/copilota/parser/typescript.py` con grammars `language_typescript()` y `language_tsx()`. Mapea `interface_declaration`, `type_alias_declaration`, `enum_declaration`. Registrado en `ParserRegistry` con extensions `.ts`, `.tsx`.
- JavaScript: quitados `.ts`, `.tsx` de `file_extensions` (ahora solo `.js`, `.jsx`, `.mjs`).
- Agregada dependencia `tree-sitter-typescript>=0.23` en pyproject.toml.

**Archivos modificados:** `src/copilota/parser/rust.py`, `src/copilota/parser/go.py`, `src/copilota/parser/javascript.py`, `src/copilota/cli.py`, `pyproject.toml`
**Archivos creados:** `src/copilota/parser/typescript.py`

---

## T8: Performance — cache Parser + batch add + list_repos

**Problema existente:**
- Cada llamada a `parse_file()` creaba un `Parser(TS_LANGUAGE)` nuevo. En un repo de 10k archivos esto es overhead medible (el objeto Parser es reutilizable).
- `add_chunks` hacía un solo `collection.add()` con todos los chunks → crash en repos grandes (límite interno de ChromaDB).
- `list_repos()` llamaba `collection.get()` sin args → cargaba **todos** los documentos + embeddings en RAM solo para contar metadatos.

**Resolución aplicada:**
- Todos los 7 parsers (python, javascript, typescript, php, go, rust, markdown-no-aplica) ahora cachean su `Parser` en un class attribute `_ts_parser` (lazy init). TypeScript cachea dos (`.ts` y `.tsx`).
- `add_chunks` ahora usa `collection.upsert()` (safe para reindex incremental) en lotes de 512.
- `list_repos()` ahora usa `collection.get(include=["metadatas"])` → solo carga metadatos, no documents ni embeddings.

**Archivos modificados:** `src/copilota/parser/python.py`, `javascript.py`, `typescript.py`, `php.py`, `go.py`, `rust.py`, `src/copilota/storage/vector_db.py`

---

## T9: CLI improvements

**Problema existente:**
- `--mock-embeddings` repetido como option en 4 commands (index, search, ask, context).
- Version hardcoded `"0.2.0"` en `@click.version_option` → drift con pyproject.toml.
- No había comando para eliminar un repo del índice.
- `asyncio.run()` en `ask` break si se ejecuta dentro de un event loop existente (taskrunner integration).

**Resolución aplicada:**
- `--mock-embeddings` movido a group option en `main()` con `@click.pass_context`. Los subcommands lo leen de `ctx.obj["mock_embeddings"]`.
- Version ahora se lee de `importlib.metadata.version("copilota")`.
- Agregado comando `delete <repo_path>` que llama `store.delete_by_repo(resolve_repo(path))`.
- `_run_async()` wrapper: detecta si hay un event loop corriendo; si no, usa `asyncio.run()`. Si sí (contexto async), delega a `anyio.from_thread.run()`.

**Archivos modificados:** `src/copilota/cli.py`

---

## T10: RAG pipeline improvements

**Problema existente:**
- Sin threshold de relevancia: chunks con score bajo (junk) se les daba al LLM igual.
- Sin budget de contexto: 5 funciones de 200 líneas podían exceder la ventana de un modelo local 27B.
- El contexto se armaba concatenando todo sin límite.

**Resolución aplicada:**
- Agregado `MIN_SCORE_THRESHOLD` (default 0.0 — desactivado por defecto para no romper mock mode; configurable para uso real).
- Agregado `MAX_CONTEXT_CHARS = 8000`: el `_build_context` ahora trunca documentos individuales y detiene la acumulación cuando se alcanza el presupuesto.
- Los resultados bajo el threshold se filtran antes de armar el contexto y no aparecen en `sources`.

**Archivos modificados:** `src/copilota/core/rag.py`

---

## Resumen de verificación

| Check | Resultado |
|-------|-----------|
| `pytest tests/` | 51 passed |
| `ruff check src/ tests/` | All checks passed |
| `./install.sh` | Instalación exitosa en `~/.local/copilota` |
| `copilota --version` | `copilota, version 0.2.0` |
| `copilota info` | 7 lenguajes, extensiones correctas, LLM openai |
| `copilota --mock-embeddings search ...` | Funciona (tabla vacía con DB limpia) |

---

## T11: Fix taskrunner RAG plugin — binario no encontrado (bug en tasks-engine)

**Problema existente:**
- El plugin RAG de taskrunner (`plugins/rag/rag.sh`) recibía `RAG_BIN` desde la sección `## rag` de `plugins.md`, que tenía `RAG_BIN: ~/bin/copilota` — una ruta inexistente.
- El `.config` global definía `COPILOTA_PATH=~/.local/copilota` pero esa variable **nunca se usaba**: el hook `before_prompt` resolvía `{config:RAG_BIN}` desde `plugins.md`, no desde `.config`.
- Resultado: las 4 tareas de prueba se ejecutaron sin contexto RAG (log: "RAG: binario no encontrado; se omite el contexto RAG" ×10).

**Resolución aplicada:**
- Corregida `plugins.md` (en `~/.local/tasks-engine/` y `~/proyectos/tasks-engine/`): `RAG_BIN: ~/.local/copilota/bin/copilota` y `ASPECTOS_DIR: ~/proyectos/aspectos`.
- Verificado que `copilota context` funciona y devuelve fragmentos indexados.
- Re-ejecutada tarea 1 para confirmar que el retrieval RAG dispara correctamente.

**Archivos modificados:** `~/.local/tasks-engine/plugins.md`, `~/proyectos/tasks-engine/plugins.md`

---

## T12: Fix agent_lib.sh — serialización de flags CLI (bug en tasks-engine)

**Problema existente:**
- `construir_flags_agente` imprimía los flags vía `printf '%s\n'` y `ejecutar_agente_cli` los leía línea por línea.
- El system prompt de 35 líneas se convertía en 35 argumentos separados → Qwen Code CLI fallaba al parsear (mostraba help, exit 1).
- Todas las tareas de taskrunner fallaban antes de ejecutar el agente.

**Resolución aplicada:**
- Reescrita `ejecutar_agente_cli` para construir un array `cmd_args` directamente con `case`, pasando `$sp_texto` como un solo elemento del array.
- Sincronizado en `~/.local/tasks-engine/agent_lib.sh` y `~/proyectos/tasks-engine/agent_lib.sh`.

**Archivos modificados:** `~/.local/tasks-engine/agent_lib.sh`, `~/proyectos/tasks-engine/agent_lib.sh`
