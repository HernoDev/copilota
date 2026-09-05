#!/bin/bash
# install.sh - Instala copilota en un venv dedicado (~/.local/copilota)
# Crea el venv si no existe, instala el paquete (no editable) y deja la
# instalacion lista para el taskrunner (RAG). Es idempotente: volvelo a
# correr para actualizar la instalacion despues de cambiar el codigo.
#
# Uso:
#   ./install.sh                        instala en ~/.local/copilota
#   COPILOTA_HOME=/ruta ./install.sh    instala en otro prefijo
#
# Requiere: python >= 3.10 y red (solo en la instalacion fresca, para
# descargar las dependencias).

# ================================
# Configuración
# ================================
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
INSTALL_DIR="${COPILOTA_HOME:-$HOME/.local/copilota}"
TASKRUNNER_DIR="$HOME/.local/tasks-engine"
ASPECTOS_DIR="$HOME/proyectos/aspectos"

# ================================
# Logging
# ================================
log() {
    echo "[install] $1"
}

log_error() {
    echo "[install] ERROR: $1" >&2
}

fallo() {
    log_error "$1"
    exit 1
}

# ================================
# Funciones
# ================================
buscar_python() {
    # Elige el python >= 3.10 mas alto disponible (solo se usa para crear el venv)
    local cand
    for cand in python3.13 python3.12 python3.11 python3.10 python3; do
        command -v "$cand" >/dev/null 2>&1 || continue
        if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            PYTHON_BIN="$cand"
            return 0
        fi
    done
    return 1
}

preparar_venv() {
    if [ -x "$INSTALL_DIR/bin/python" ]; then
        log "Venv existente: $INSTALL_DIR (se reutiliza)"
        return 0
    fi
    if [ -e "$INSTALL_DIR" ]; then
        fallo "$INSTALL_DIR existe pero no es un venv (no hay bin/python). Revisalo a mano antes de continuar."
    fi
    buscar_python || fallo "No hay python >= 3.10 en el PATH (verificá: python3 --version)."
    log "Creando venv en $INSTALL_DIR con $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"
    "$PYTHON_BIN" -m venv "$INSTALL_DIR" || fallo "No se pudo crear el venv."
}

instalar_paquete() {
    log "Instalando copilota desde $SCRIPT_DIR (instalacion normal, no editable)..."
    "$INSTALL_DIR/bin/pip" install --quiet "$SCRIPT_DIR" \
        || fallo "pip falló instalando copilota (¿faltan dependencias? ¿red?)."
}

verificar() {
    log "Verificando instalacion..."
    "$INSTALL_DIR/bin/copilota" --help >/dev/null 2>&1 \
        || fallo "bin/copilota no responde tras la instalacion."
    local version
    version="$("$INSTALL_DIR/bin/python" -c 'import importlib.metadata as m; print(m.version("copilota"))')"
    log "copilota $version listo en $INSTALL_DIR/bin/copilota"
}

integrar_taskrunner() {
    if [ ! -d "$TASKRUNNER_DIR" ]; then
        log "No hay taskrunner en $TASKRUNNER_DIR; se omite la integracion."
        return 0
    fi

    # Forma con ~ si esta bajo $HOME (el taskrunner expande el ~ inicial)
    local path_config
    case "$INSTALL_DIR" in
        "$HOME"/*) path_config="~${INSTALL_DIR#"$HOME"}" ;;
        *) path_config="$INSTALL_DIR" ;;
    esac

    local config="$TASKRUNNER_DIR/.config"

    if [ ! -f "$config" ]; then
        local aspectos=""
        [ -d "$ASPECTOS_DIR" ] && aspectos="$ASPECTOS_DIR"
        cat > "$config" <<EOF
# Config global de taskrunner (RAG / copilota)
# Formato KEY=value, editable a mano. Un # al inicio de linea es comentario.

# Ruta de instalacion de copilota (prefijo con bin/, o directorio con el binario).
# Vacio = RAG desactivado.
COPILOTA_PATH=$path_config

# Repo de aspectos (implementaciones reutilizables). Vacio = desactivado.
ASPECTOS_DIR=$aspectos

# Modo RAG por defecto cuando el proyecto no define .task_system/.rag_mode:
# off | aspectos | aspectos_codigo
RAG_DEFAULT_MODE=off

# top-k de retrieval
RAG_K_ASPECTOS=5
RAG_K_CODE=5
EOF
        log "Config RAG creada en $config (modo por defecto: off)"
        return 0
    fi

    if ! grep -q '^COPILOTA_PATH=' "$config"; then
        echo "COPILOTA_PATH=$path_config" >> "$config"
        log "COPILOTA_PATH agregado a $config"
        return 0
    fi

    local actual
    actual="$(grep -m1 '^COPILOTA_PATH=' "$config" | cut -d= -f2-)"
    local actual_exp
    case "$actual" in
        "~"/*) actual_exp="$HOME${actual#"~"}" ;;
        "~") actual_exp="$HOME" ;;
        *) actual_exp="$actual" ;;
    esac

    if [ -z "$actual" ]; then
        sed -i "s|^COPILOTA_PATH=.*|COPILOTA_PATH=$path_config|" "$config"
        log "COPILOTA_PATH estaba vacio; ahora apunta a $path_config"
    elif [ "$actual_exp" = "$INSTALL_DIR" ]; then
        log "COPILOTA_PATH ya apunta a esta instalacion ($actual)"
    else
        log "AVISO: COPILOTA_PATH apunta a '$actual'; no se modifica. Si querés usar $path_config, editalo a mano en $config"
    fi
}

resumen() {
    echo ""
    log "========================================="
    log "  Instalación completada"
    log "========================================="
    log "Binario:    $INSTALL_DIR/bin/copilota"
    log "Base datos: ~/.local/share/copilota (ChromaDB persistente)"
    echo ""
    log "Próximos pasos:"
    log "  # Indexar el repo de aspectos (y re-indexar cuando agregues aspectos):"
    log "  $INSTALL_DIR/bin/copilota index $ASPECTOS_DIR"
    log "  # Indexar un proyecto (el reindex es limpio, reemplaza el namespace):"
    log "  $INSTALL_DIR/bin/copilota index ~/proyectos/<proyecto>"
    log "  # Ver estado de los índices:"
    log "  $INSTALL_DIR/bin/copilota info"
    if [ -d "$TASKRUNNER_DIR" ]; then
        log "  # Ver la config RAG del taskrunner:"
        log "  taskrunner status"
    fi
    echo ""
    log "Para actualizar tras cambiar el codigo de copilota, volvé a correr: $SCRIPT_DIR/install.sh"
}

# ================================
# Main
# ================================
preparar_venv
instalar_paquete
verificar
integrar_taskrunner
resumen
