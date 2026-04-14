"""Tests para módulos core: embedder, retriever, rag."""

from textwrap import dedent
from pathlib import Path

import pytest
import pytest_asyncio

from copilota.core.embedder import EmbeddingModel
from copilota.core.retriever import Retriever
from copilota.core.rag import RAGPipeline
from copilota.llm.ollama import OllamaLLM
from copilota.storage.vector_db import VectorStore


@pytest.fixture
def embedder():
    return EmbeddingModel(use_mock=True)


@pytest.fixture
def store():
    s = VectorStore()
    s.clear()
    yield s
    try:
        s.clear()
    except Exception:
        pass


@pytest.fixture
def retriever(store, embedder):
    return Retriever(store, embedder)


@pytest.fixture
def llm():
    return OllamaLLM()


@pytest.fixture
def rag(retriever, llm):
    return RAGPipeline(retriever, llm)


def _mock_embedding(dim=384):
    return [0.1] * dim


class TestEmbeddingModel:
    def test_encode_returns_list_of_lists(self, embedder):
        result = embedder.encode(["hello", "world"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], list)

    def test_encode_single_returns_list(self, embedder):
        result = embedder.encode_single("hello")
        assert isinstance(result, list)
        assert len(result) > 0


class TestRetriever:
    def test_search_returns_results_after_adding(self, retriever, store, embedder):
        meta = {"filepath": "test.py", "language": "python", "node_type": "function", "name": "hello", "start_line": 1, "end_line": 5}
        store._collection.add(
            ids=["test1"],
            embeddings=[_mock_embedding()],
            documents=["def hello(): pass"],
            metadatas=[meta],
        )
        results = retriever.search("hello function", top_k=5)
        assert len(results) >= 1
        assert results[0].name == "hello"


@pytest.mark.asyncio
class TestRAGPipeline:
    async def test_query_returns_answer_and_sources(self, rag, store, embedder):
        meta = {"filepath": "auth.py", "language": "python", "node_type": "function", "name": "login", "start_line": 10, "end_line": 20}
        store._collection.add(
            ids=["rag1"],
            embeddings=[_mock_embedding()],
            documents=["def login(user, password): ..."],
            metadatas=[meta],
        )
        result = await rag.query("¿Cómo funciona el login?")
        assert "answer" in result
        assert "sources" in result
        assert len(result["sources"]) >= 1
