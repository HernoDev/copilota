# Continuar: copilota + taskrunner — estado y pendientes

Handoff actualizado. **Fecha: 2026-09-05.**

La integración RAG quedó implementada y testeada el 2026-09-03.
El 2026-09-04 se agregó el provider OpenAI-compatible (llama-swap) y los flags
`--model-select` / subcomando `models`.

**Nuevos hallazgos (2026-09-05):**
1. El instalador de copilota **no agrega** `~/.local/copilota/bin` al PATH de
   `~/.bashrc` → el comando `copilota` no está disponible desde cualquier
   directorio.
2. **No existe** subcomando `copilota config` para ver/cambiar la configuración
   desde el CLI (el usuario lo pidió, análogo a `--model-list` del taskrunner).
3. **CRÍTICO:** el código RAG del taskrunner **NO está backportado** al source
   de verdad (`~/proyectos/tasks-engine/`). Solo existe en la instalación
   (`~/.local/tasks-engine/`) y en el mirror viejo (`~/bash-utils/IA/tasks-engine/`).
   Si se corre el instalador desde `~/proyectos/tasks-engine/`, se **pierde**
   toda la integración RAG.
4. El mirror `~/bash-utils/IA/tasks-engine/` sigue existiendo y es **redundante**
   (idéntico al instalado). El source de verdad ya es `~/proyectos/tasks-engine/`.

---

## 1. Estado verificado (2026-09-05)

### 1.1 Copilota — instalación

| Item | Estado |
|---|---|
| Binario | `~/.local/copilota/bin/copilota` → v0.2.0, responde |
| PATH en `~/.bashrc` | **AUSENTE** — no hay línea con `~/.local/copilota/bin` |
| Symlink `~/.local/bin/copilota` | **NO existe** |
| Instalación editable | No (venv dedicado, `pip install` no editable) |
| Re-instalación idempotente | Sí — `cd ~/proyectos/copilota && ./install.sh` |
| `~/.copilota/config.yaml` | provider=openai, model=qwen3.8-27b-largo, base_url=http://santollama, port=8080, timeout=300 |

**Problema:** el usuario no puede invocar `copilota` sin ruta completa. El
instalador de copilota (`install.sh`) no tiene la función de agregar al PATH
ni de crear el symlink. El instalador del taskrunner SÍ tiene esa lógica
(`configurar_path` que pregunta y agrega a `.bashrc`), pero no la aplica a
copilota.

### 1.2 Copilota — CLI

| Subcomando | Existe | Notas |
|---|---|---|
| `index` | Sí | Indexa repo en ChromaDB |
| `search` | Sí | Búsqueda por vector |
| `ask` | Sí | RAG + LLM, acepta `--model-select N` y `--model NAME` |
| `context` | Sí | Retrieval solo (lo usa taskrunner) |
| `models` | Sí | Lista modelos del servidor LLM |
| `info` | Sí | Muestra config + stats |
| `config` | **NO** | **Falta** — ver/cambiar parámetros de config desde CLI |

**Falta:** un subcomando `copilota config` que permita:
- Ver la config actual (como `copilota info` pero centrado en config)
- Cambiar parámetros individualmente (provider, model, base_url, port, etc.)
- Guardar en `~/.copilota/config.yaml`

Análogo a `taskrunner --model-list` / `--model-select N` pero para la config
completa de copilota.

### 1.3 Taskrunner — source vs instalado

**Source de verdad:** `~/proyectos/tasks-engine/`
**Instalado:** `~/.local/tasks-engine/`
**Mirror viejo (redundante):** `~/bash-utils/IA/tasks-engine/`

| Archivo | Source (`~/proyectos/`) | Instalado (`~/.local/`) | Mirror (`~/bash-utils/`) |
|---|---|---|---|
| `taskrunner.sh` | **SIN RAG** | Con `preguntar_modo_rag` + sección RAG en status | Con RAG (== instalado) |
| `ejecutor_tarea.sh` | **SIN RAG** | Con sección RAG completa (líneas 302-426) | Con RAG (== instalado) |
| `install.sh` | **SIN RAG** | Con `configurar_rag` (setup interactivo) | Con RAG (== instalado) |
| `.config` | N/A (se crea en install) | Existe: COPILOTA_PATH, ASPECTOS_DIR, RAG_DEFAULT_MODE=aspectos | N/A |

**Código RAG que falta en el source (`~/proyectos/tasks-engine/`):**

En `ejecutor_tarea.sh`:
- `CONFIG_GLOBAL` variable
- `leer_config_global()` — lee KEY=value de `.config`
- `resolver_bin_copilota()` — encuentra binario de copilota
- `resolver_modo_rag()` — precedencia proyecto > global > off
- `extraer_descripcion_tarea()` — awk sobre `## Descripción`
- `construir_contexto_rag()` — llama a `copilota context`, arma RAG_CONTEXTO
- Inyección de RAG_CONTEXTO en `ejecutar_tarea()`

En `taskrunner.sh`:
- `preguntar_modo_rag()` — al final de `cmd_analyze`
- Sección RAG en `cmd_status`

En `install.sh`:
- `configurar_rag()` — setup interactivo (¿usar RAG? → ruta → aspectos dir)

### 1.4 Índices y datos

| Repo | Estado |
|---|---|
| `~/proyectos/aspectos/` | Indexado: 753 chunks / 100 archivos |
| `~/proyectos/buscarag/` | Indexado: 95 chunks / 19 archivos. Modo `aspectos`. Tareas PENDIENTE (step 0). |

### 1.5 Provider OpenAI (llama-swap)

- `src/copilota/llm/openai_compat.py` → `OpenAICompatibleLLM` (POST `/v1/chat/completions`)
- `src/copilota/llm/factory.py` → rutea `openai`/`openai_compatible` → `OpenAICompatibleLLM`
- `copilota models` → lista 9 modelos de llama-swap
- `copilota ask` → funciona (requiere llama-swap libre, timeout 300s)
- `ruff check src/` → limpio. `pytest tests/ -q` → 43 passed.

---

## 2. Qué queda por hacer (en orden de prioridad)

### 2.1 BACKPORT RAG AL SOURCE DE TASKRUNNER (CRÍTICO)

Copiar el código RAG desde `~/.local/tasks-engine/` (o `~/bash-utils/IA/tasks-engine/`,
que son idénticos) a `~/proyectos/tasks-engine/`:

1. **`ejecutor_tarea.sh`**: agregar la sección RAG completa (funciones + inyección)
2. **`taskrunner.sh`**: agregar `preguntar_modo_rag` + sección RAG en `cmd_status`
3. **`install.sh`**: agregar `configurar_rag` (setup interactivo de RAG)

Luego correr el instalador desde el source para verificar:
```bash
cd ~/proyectos/tasks-engine && ./install.sh
```

### 2.2 AGREGAR PATH DE COPILOTA AL INSTALADOR

En `~/proyectos/copilota/install.sh`, agregar una función que:
1. Verifique si `~/.local/copilota/bin` ya está en `~/.bashrc`
2. Si no está, agregue `export PATH="$HOME/.local/copilota/bin:$PATH"`
3. Opcionalmente: cree symlink `~/.local/bin/copilota` → `~/.local/copilota/bin/copilota`

Análogo a lo que hace `configurar_path` en el instalador de taskrunner.

### 2.3 AGREGAR SUBCOMANDO `copilota config`

En `src/copilota/cli.py`, agregar subcomando `config` que permita:
- `copilota config` → muestra la config actual (parámetro por parámetro)
- `copilota config --set model qwen3.8-27b-largo` → cambia un parámetro
- `copilota config --set provider ollama` → cambia provider
- `copilota config --list` → lista todos los parámetros disponibles
- Guardar en `~/.copilota/config.yaml` (crear el directorio si no existe)

Parámetros configurables: `enabled`, `provider`, `model`, `base_url`, `port`,
`api_path`, `chat_api_path`, `temperature`, `max_tokens`, `timeout`.

### 2.4 LIMPIAR MIRROR REDUNDANTE (baja prioridad)

`~/bash-utils/IA/tasks-engine/` es redundante ahora que el source de verdad
es `~/proyectos/tasks-engine/`. Opciones:
- Borrarlo (ya está en git)
- Dejarlo como backup temporal
- Poner un README indicando que el source se movió

### 2.5 PRUEBA FINAL DE `copilota ask` (requiere llama-swap libre)

```bash
~/.local/copilota/bin/copilota ask "qué es un aspecto" -r ~/proyectos/aspectos -k 4
```

### 2.6 PRIMER TEST REAL DE RAG EN TASKRUNNER

```bash
cd ~/proyectos/buscarag
taskrunner start
```

### 2.7 MEDIR VALOR DEL RAG

Comparar con y sin RAG (tiempo, calidad, si el modelo re-derivó lo del aspecto).

---

## 3. Decisiones de diseño (acordadas, mantener)

- **Empezar con `aspectos` solo**; `aspectos_codigo` existe y está testeable.
- Los aspectos viven en **otro repo** (`~/proyectos/aspectos`), independiente.
- La config de RAG se captura en el **install.sh de taskrunner** (no en el de copilota).
- **Reindex manual** (el ejecutor NO re-indexa solo).
- **No fatal por diseño**: cualquier fallo de RAG loguea y la tarea sigue.
- **Provider LLM**: `openai` (OpenAI-compatible) → llama-swap. Modelo default:
  `qwen3.8-27b-largo`. Se puede cambiar con `--model-select N` o `--model NAME`.
- **Source de verdad de taskrunner**: `~/proyectos/tasks-engine/` (no más bash-utils).
- **Convención de instalación**: siempre modificar en el source y luego correr
  el instalador.

## 4. Convenciones a respetar

- Código/identificadores en **inglés**; mensajes, `--help` y docs en
  **castellano** (voseo).
- Formato de log del executor: `[$(date)] [TX:..] [TAREA] msg`.
- **Source de verdad**: `~/proyectos/tasks-engine/` (ya NO es bash-utils).
- Sin subagentes (GPU compartida). Explicar antes de operar.
- El instalador es idempotente: se puede correr varias veces.

## 5. Comandos de referencia

```bash
C=~/.local/copilota/bin/copilota

# Índices
$C info
$C index ~/proyectos/aspectos
$C index ~/proyectos/buscarag

# Modelos del servidor LLM
$C models

# Retrieval manual (lo mismo que inyecta el ejecutor)
$C context "cola de envío de correos" -r ~/proyectos/aspectos -k 5

# Pregunta RAG (usa LLM — requiere llama-swap libre)
$C ask "cómo funciona la cola de emails" -r ~/proyectos/aspectos -k 4
$C ask "..." -r ~/proyectos/aspectos -k 4 --model-select 6

# Config RAG (taskrunner)
cat ~/.local/tasks-engine/.config
cat ~/proyectos/buscarag/.task_system/.rag_mode   # off | aspectos | aspectos_codigo

# Logs RAG del ejecutor
grep 'RAG:' ~/proyectos/buscarag/.task_system/.task_executor.log

# Actualizar la instalación de copilota
cd ~/proyectos/copilota && ./install.sh

# Actualizar la instalación de taskrunner (DESDE EL SOURCE)
cd ~/proyectos/tasks-engine && ./install.sh
```
