"""Tests para módulos core: embedder, indexer, retriever, rag."""

from pathlib import Path

import git
import pytest

from copilota.core.embedder import EmbeddingModel
from copilota.core.indexer import Indexer
from copilota.core.rag import RAGPipeline
from copilota.core.retriever import Retriever
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
        meta = {
            "filepath": "test.py", "language": "python", "node_type": "function",
            "name": "hello", "start_line": 1, "end_line": 5,
        }
        store._collection.add(
            ids=["test1"],
            embeddings=[_mock_embedding()],
            documents=["def hello(): pass"],
            metadatas=[meta],
        )
        results = retriever.search("hello function", top_k=5)
        assert len(results) >= 1
        assert results[0].name == "hello"

    def test_search_repo_filter(self, retriever, store, embedder):
        meta_a = {
            "filepath": "a.py", "language": "python", "node_type": "function",
            "name": "fa", "repo": "/r/a",
        }
        meta_b = {
            "filepath": "b.py", "language": "python", "node_type": "function",
            "name": "fb", "repo": "/r/b",
        }
        store._collection.add(
            ids=["ra", "rb"],
            embeddings=[_mock_embedding(), _mock_embedding()],
            documents=["def fa(): pass", "def fb(): pass"],
            metadatas=[meta_a, meta_b],
        )
        results = retriever.search("f", top_k=5, repo="/r/a")
        assert len(results) == 1
        assert results[0].name == "fa"


@pytest.mark.asyncio
class TestRAGPipeline:
    async def test_query_returns_answer_and_sources(self, rag, store, embedder):
        meta = {
            "filepath": "auth.py", "language": "python", "node_type": "function",
            "name": "login", "start_line": 10, "end_line": 20,
        }
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


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = git.Repo.init(str(tmp_path))
    (tmp_path / "app.py").write_text("def greet(name):\n    return f'hi {name}'\n")
    repo.git.add("app.py")
    repo.index.commit("init", author=git.Actor("t", "t@t"), committer=git.Actor("t", "t@t"))
    return tmp_path


@pytest.fixture
def tmp_store(tmp_path: Path) -> VectorStore:
    return VectorStore(persist_directory=tmp_path / "chroma")


class TestIndexer:
    def test_index_repo_returns_counts(self, git_repo, tmp_store, embedder):
        result = Indexer(tmp_store, embedder).index_repo(git_repo)
        assert result.files == 1
        assert result.chunks >= 1
        assert result.repo_id == str(git_repo.resolve())

    def test_chunks_have_repo_and_relative_path(self, git_repo, tmp_store, embedder):
        Indexer(tmp_store, embedder).index_repo(git_repo)
        data = tmp_store._collection.get()
        meta = data["metadatas"][0]
        assert meta["repo"] == str(git_repo.resolve())
        assert meta["filepath"] == "app.py"

    def test_reindex_removes_stale_files(self, git_repo, tmp_store, embedder):
        indexer = Indexer(tmp_store, embedder)
        assert indexer.index_repo(git_repo).chunks >= 1
        (git_repo / "app.py").unlink()
        result = indexer.index_repo(git_repo)
        assert result.chunks == 0
        assert tmp_store.count() == 0

    def test_untracked_files_are_indexed(self, git_repo, tmp_store, embedder):
        (git_repo / "new.py").write_text("def extra():\n    return 42\n")
        result = Indexer(tmp_store, embedder).index_repo(git_repo)
        assert result.files == 2
