"""Stub de Ollama: implementación placeholder hasta tener Ollama disponible."""

from __future__ import annotations

from copilota.llm.base import BaseLLM


class OllamaLLM(BaseLLM):
    """Stub que simula respuestas de Ollama. Reemplazar cuando Ollama esté disponible."""

    def __init__(self, model: str = "qwen2.5-coder", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        return (
            f"[Ollama stub - model: {self.model}]\n"
            f"Prompt recibido ({len(prompt)} chars).\n"
            f"Esta es una respuesta de prueba. "
            f"Cuando Ollama esté configurado, esta respuesta vendrá del modelo local."
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        last_msg = messages[-1].get("content", "") if messages else ""
        return (
            f"[Ollama stub - model: {self.model}]\n"
            f"Chat: último mensaje tiene {len(last_msg)} chars.\n"
            f"Conectar Ollama real para obtener respuestas inteligentes."
        )
