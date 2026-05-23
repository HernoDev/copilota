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
cd /home/santollama/proyectos/copilota
source venv/bin/activate
python -m copilota.cli index /path/to/repo
```

### 2. Hacer Preguntas (RAG)
```bash
python -m copilota.cli ask "tu pregunta sobre el código" -c config/default.yaml
```

### 3. Buscar Código
```bash
python -m copilota.cli search "término de búsqueda" -k 5
```

### 4. Ver Información
```bash
python -m copilota.cli info
```

## Ejemplo de Uso

```bash
# Indexar el propio repositorio
python -m copilota.cli index /home/santollama/proyectos/copilota

# Preguntar sobre los parsers
python -m copilota.cli ask "¿Qué lenguajes de programación soporta el parser?" -c config/default.yaml
```

## Configuración YAML

Archivo: `config/default.yaml`

```yaml
llm:
  enabled: true
  provider: ollama
  model: qwen35:27b_64k
  base_url: http://localhost
  port: 11434
  api_path: /api/generate
  chat_api_path: /api/chat
  temperature: 0.7
  max_tokens: 2048
  timeout: 120
```

## Notas Importantes

1. **Ollama debe estar corriendo** antes de usar los comandos
2. **El modelo debe estar cargado** (puede tardar la primera vez)
3. **Los datos persisten** en `~/.local/share/copilota`
4. **Reindexar** si hay cambios en el código fuente

## Lenguajes Soportados

- Python
- JavaScript
- PHP
- Go
- Rust
