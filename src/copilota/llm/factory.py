"""Factory para crear el proveedor LLM correcto según configuración."""

from __future__ import annotations

from copilota.config import AppConfig
from copilota.llm.base import BaseLLM
from copilota.llm.ollama import OllamaLLM as OllamaStub
from copilota.llm.ollama_real import OllamaLLM


def create_llm(config: AppConfig) -> BaseLLM:
    if not config.llm.enabled:
        return OllamaStub()
    if config.llm.provider == "ollama":
        return OllamaLLM(config.llm)
    raise ValueError(f"Proveedor LLM no soportado: {config.llm.provider}")
