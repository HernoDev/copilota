"""Proveedor OpenAI-compatible: llama-swap, vLLM, llama.cpp server."""

from __future__ import annotations

import httpx

from copilota.config import LLMConfig
from copilota.llm.base import BaseLLM


class OpenAICompatibleLLM(BaseLLM):
    """Cliente para cualquier API compatible con OpenAI (/v1/chat/completions)."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=10.0, read=float(self.config.timeout), write=10.0, pool=10.0
        )

    def _payload(
        self,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict:
        return {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }

    async def _post(self, payload: dict) -> str:
        async with httpx.AsyncClient(timeout=self._timeout()) as client:
            resp = await client.post(self.config.chat_url, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self._post(self._payload(messages, temperature, max_tokens))

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return await self._post(self._payload(messages, temperature, max_tokens))
