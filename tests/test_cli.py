"""Tests para la CLI (comando context)."""

import pytest
from click.testing import CliRunner

from copilota.cli import main
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


def _seed(store: VectorStore):
    meta = {
        "filepath": "src/auth.py",
        "language": "python",
        "node_type": "function",
        "name": "send_welcome",
        "repo": "/repo/one",
        "start_line": 1,
        "end_line": 3,
    }
    store._collection.add(
        ids=["cli1"],
        embeddings=[[0.1] * 384],
        documents=["def send_welcome(user):\n    mailer.send(user, 'welcome')"],
        metadatas=[meta],
    )


class TestContextCommand:
    def test_context_prints_full_code(self, store):
        _seed(store)
        runner = CliRunner()
        result = runner.invoke(main, ["context", "welcome email", "--mock-embeddings", "-k", "3"])
        assert result.exit_code == 0
        assert "send_welcome" in result.output
        assert "mailer.send" in result.output
        assert "Fragmentos: 1" in result.output

    def test_context_repo_filter_excludes_other_repos(self, store):
        _seed(store)
        runner = CliRunner()
        args = ["context", "welcome email", "--mock-embeddings", "-r", "/repo/other"]
        result = runner.invoke(main, args)
        assert result.exit_code == 0
        assert "Fragmentos: 0" in result.output
