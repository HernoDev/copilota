"""Indexador: orquesta parsing, chunking y embedding para almacenar código en la vector DB."""

from __future__ import annotations

import hashlib
from pathlib import Path

import git

from copilota.core.embedder import EmbeddingModel
from copilota.parser.registry import ParserRegistry
from copilota.storage.models import ASTNode, CodeChunk, NodeType
from copilota.storage.vector_db import VectorStore


class Indexer:
    """Indexa repositorios Git en la vector DB."""

    def __init__(self, vector_store: VectorStore, embedder: EmbeddingModel):
        self._store = vector_store
        self._embedder = embedder

    def index_repo(self, repo_path: str | Path) -> int:
        repo_path = Path(repo_path)
        repo = git.Repo(str(repo_path))
        total_chunks = 0

        for filepath in self._iter_tracked_files(repo):
            if not ParserRegistry.has_parser_for_file(filepath):
                continue
            try:
                chunks = self._index_file(repo_path / filepath)
                total_chunks += chunks
            except Exception:
                continue

        return total_chunks

    def _iter_tracked_files(self, repo: git.Repo):
        for blob in repo.head.commit.tree.traverse():
            if blob.type == "blob":
                yield Path(blob.path)

    def _index_file(self, filepath: Path) -> int:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
        parser = ParserRegistry.get_for_file(filepath)

        nodes = parser.parse_file(filepath, source)
        if not nodes:
            return 0

        self._store.delete_by_filepath(str(filepath))

        chunks = self._create_chunks(nodes)
        if not chunks:
            return 0

        texts = [c.embedding_text for c in chunks]
        embeddings = self._embedder.encode(texts)
        self._store.add_chunks(chunks, embeddings)
        return len(chunks)

    def _create_chunks(self, nodes: list[ASTNode]) -> list[CodeChunk]:
        chunks = []
        for node in nodes:
            if node.node_type in (NodeType.FUNCTION, NodeType.METHOD, NodeType.CLASS, NodeType.INTERFACE, NodeType.STRUCT, NodeType.TRAIT, NodeType.ENUM):
                parser = ParserRegistry.get_for_language(node.language)
                chunk_text = parser.get_chunk_text(node)
                chunk_id = self._make_chunk_id(node)
                chunks.append(
                    CodeChunk(
                        id=chunk_id,
                        node=node,
                        embedding_text=chunk_text,
                    )
                )
        return chunks

    @staticmethod
    def _make_chunk_id(node: ASTNode) -> str:
        raw = f"{node.filepath}:{node.start_line}:{node.name}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
