"""Tests para los clientes LLM reales usando httpx.MockTransport."""

import json

import httpx
import pytest

from copilota.config import LLMConfig
from copilota.llm.ollama_real import OllamaLLM
from copilota.llm.openai_compat import OpenAICompatibleLLM


def _make_openai_llm(base_url="http://localhost", port=8080, model="test-model"):
    cfg = LLMConfig(
        enabled=True,
        provider="openai",
        model=model,
        base_url=base_url,
        port=port,
        chat_api_path="/v1/chat/completions",
        timeout=30,
    )
    return OpenAICompatibleLLM(cfg)


def _make_ollama_llm(model="qwen2.5-coder"):
    cfg = LLMConfig(
        enabled=True,
        provider="ollama",
        model=model,
        base_url="http://localhost",
        port=11434,
        timeout=30,
    )
    return OllamaLLM(cfg)


class TestOpenAICompatGenerate:
    def test_success(self):
        llm = _make_openai_llm()

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["model"] == "test-model"
            assert body["messages"][0]["role"] == "system"
            assert body["messages"][1]["role"] == "user"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "Hola mundo"}}]},
            )

        async def run():
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                llm._client = client
                return await llm.generate("hola", system_prompt="eres un bot")

        result = _run_async(run())
        assert result == "Hola mundo"

    def test_content_null(self):
        llm = _make_openai_llm()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": None}}]},
            )

        async def run():
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                llm._client = client
                return await llm.generate("hola")

        result = _run_async(run())
        assert result is None or result == ""

    def test_missing_choices(self):
        llm = _make_openai_llm()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        async def run():
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                llm._client = client
                return await llm.generate("hola")

        result = _run_async(run())
        assert result == ""

    def test_http_error(self):
        llm = _make_openai_llm()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        async def run():
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                llm._client = client
                return await llm.generate("hola")

        with pytest.raises(httpx.HTTPStatusError):
            _run_async(run())


class TestOllamaRealGenerate:
    def test_success(self):
        llm = _make_ollama_llm()

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["model"] == "qwen2.5-coder"
            assert body["stream"] is False
            return httpx.Response(200, json={"response": "respuesta ollama"})

        async def run():
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                llm._client = client
                return await llm.generate("hola")

        result = _run_async(run())
        assert result == "respuesta ollama"

    def test_response_key_missing(self):
        llm = _make_ollama_llm()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"error": "model not found"})

        async def run():
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                llm._client = client
                return await llm.generate("hola")

        result = _run_async(run())
        assert result == ""

    def test_chat_success(self):
        llm = _make_ollama_llm()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "chat response"}},
            )

        async def run():
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                llm._client = client
                return await llm.chat([{"role": "user", "content": "hi"}])

        result = _run_async(run())
        assert result == "chat response"

    def test_chat_content_null(self):
        llm = _make_ollama_llm()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": None}},
            )

        async def run():
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                llm._client = client
                return await llm.chat([{"role": "user", "content": "hi"}])

        result = _run_async(run())
        assert result is None or result == ""


def _run_async(coro):
    import asyncio

    return asyncio.get_event_loop().run_until_complete(coro)
