"""Tests para storage y vector DB."""

from pathlib import Path

import pytest

from copilota.storage.models import ASTNode, CodeChunk, NodeType
from copilota.storage.vector_db import VectorStore


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
def sample_node():
    return ASTNode(
        node_type=NodeType.FUNCTION,
        name="hello",
        source_code='def hello():\n    return "world"',
        start_line=1,
        end_line=2,
        filepath="test.py",
        language="python",
    )


@pytest.fixture
def sample_chunk(sample_node):
    return CodeChunk(
        id="test123",
        node=sample_node,
        embedding_text="function hello returns world",
    )


class TestModels:
    def test_ast_node_signature(self, sample_node):
        assert sample_node.signature == "test.py:1-2:hello"

    def test_ast_node_to_dict(self, sample_node):
        d = sample_node.to_dict()
        assert d["name"] == "hello"
        assert d["node_type"] == "function"
        assert d["filepath"] == "test.py"

    def test_code_chunk_chroma_metadata(self, sample_chunk):
        meta = sample_chunk.to_chroma_metadata()
        assert meta["filepath"] == "test.py"
        assert meta["language"] == "python"
        assert meta["node_type"] == "function"
        assert meta["name"] == "hello"


class TestVectorStore:
    def test_add_and_count(self, store, sample_chunk):
        mock_embedding = [0.1] * 384
        store.add_chunks([sample_chunk], [mock_embedding])
        assert store.count() == 1

    def test_query_returns_results(self, store, sample_chunk):
        mock_embedding = [0.1] * 384
        store.add_chunks([sample_chunk], [mock_embedding])
        results = store.query(mock_embedding, top_k=5)
        assert len(results) == 1
        assert results[0]["id"] == "test123"

    def test_delete_by_filepath(self, store, sample_chunk):
        mock_embedding = [0.1] * 384
        store.add_chunks([sample_chunk], [mock_embedding])
        assert store.count() == 1
        store.delete_by_filepath("test.py")
        assert store.count() == 0

    def test_clear(self, store, sample_chunk):
        mock_embedding = [0.1] * 384
        store.add_chunks([sample_chunk], [mock_embedding])
        store.clear()
        assert store.count() == 0
