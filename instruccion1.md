# Copilota - Instrucciones de Uso

## Configuración Actual

### Ollama Local
- **Modelo**: `qwen35:27b_64k`
- **URL**: `http://localhost:11434`
- **Estado**: ✅ Funcionando

### Base de Datos
- **Ubicación**: `~/.local/share/copilota`
- **Tipo**: ChromaDB persistente
- **Colección**: `copilota_code`

## Comandos

### 1. Indexar Repositorio
```bash
cd /path/to/repo
copilota index .
```

### 2. Hacer Preguntas (RAG)
```bash
copilota ask "tu pregunta sobre el código"
```

### 3. Buscar Código
```bash
copilota search "término de búsqueda" -k 5
```

### 4. Ver Información
```bash
copilota info
```

## Configuración Global

La configuración se carga automáticamente desde:
1. `config/default.yaml` en el directorio actual
2. `~/.copilota/config.yaml` (global)
3. Archivo especificado con `-c`

**Configuración global creada**: `~/.copilota/config.yaml`

```yaml
llm:
  enabled: true
  provider: ollama
  model: qwen35:27b_64k
  base_url: http://localhost
  port: 11434
  temperature: 0.7
  max_tokens: 2048
  timeout: 120
```

## Ejemplo de Uso

```bash
# Indexar el repositorio guia
cd /home/santollama/proyectos/guia
copilota index .

# Preguntar sobre el código
copilota ask "donde esta el email enviado al usuario"
```

## Notas Importantes

1. **Ollama debe estar corriendo** antes de usar los comandos
2. **El modelo debe estar cargado** (puede tardar la primera vez)
3. **Los datos persisten** en `~/.local/share/copilota`
4. **Reindexar** si hay cambios en el código fuente
5. **Configuración automática**: ya no necesitas `-c config/default.yaml`

## Lenguajes Soportados

- Python
- JavaScript
- PHP
- Go
- Rust
