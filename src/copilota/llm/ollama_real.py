"""Proveedor Ollama: implementación real con HTTP."""

from __future__ import annotations

from copilota.config import LLMConfig
from copilota.llm.base import BaseLLM


class OllamaLLM(BaseLLM):
    """Implementación real que llama a la API de Ollama via HTTP."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        import httpx

        payload: dict = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.config.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.config.max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        timeout = httpx.Timeout(
            connect=10.0, read=float(self.config.timeout), write=10.0, pool=10.0
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                self.config.generate_url,
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["response"]

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        import httpx

        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.config.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.config.max_tokens,
            },
        }

        timeout = httpx.Timeout(
            connect=10.0, read=float(self.config.timeout), write=10.0, pool=10.0
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                self.config.chat_url,
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
