"""Indexador: orquesta parsing, chunking y embedding para almacenar código en la vector DB."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import git

from copilota.core.embedder import EmbeddingModel
from copilota.parser.registry import ParserRegistry
from copilota.storage.models import ASTNode, CodeChunk, NodeType
from copilota.storage.vector_db import VectorStore

CHUNKED_NODE_TYPES = (
    NodeType.FUNCTION,
    NodeType.METHOD,
    NodeType.CLASS,
    NodeType.INTERFACE,
    NodeType.STRUCT,
    NodeType.TRAIT,
    NodeType.ENUM,
)


@dataclass
class IndexResult:
    repo_id: str
    files: int
    chunks: int


class Indexer:
    """Indexa repositorios Git en la vector DB, con un namespace por repo."""

    def __init__(self, vector_store: VectorStore, embedder: EmbeddingModel):
        self._store = vector_store
        self._embedder = embedder

    def index_repo(self, repo_path: str | Path) -> IndexResult:
        repo = git.Repo(str(Path(repo_path).resolve()), search_parent_directories=True)
        repo_path = Path(repo.working_tree_dir)
        repo_id = str(repo_path)

        self._store.delete_by_repo(repo_id)

        files = 0
        total_chunks = 0
        for rel_path in self._iter_files(repo, repo_path):
            if not ParserRegistry.has_parser_for_file(rel_path):
                continue
            try:
                total_chunks += self._index_file(repo_path, rel_path, repo_id)
                files += 1
            except Exception:
                continue

        return IndexResult(repo_id=repo_id, files=files, chunks=total_chunks)

    def _iter_files(self, repo: git.Repo, repo_path: Path):
        raw = repo.git.ls_files("--cached", "--others", "--exclude-standard")
        for rel in raw.splitlines():
            rel_path = Path(rel)
            if (repo_path / rel_path).is_file():
                yield rel_path

    def _index_file(self, repo_path: Path, rel_path: Path, repo_id: str) -> int:
        source = (repo_path / rel_path).read_text(encoding="utf-8", errors="ignore")
        parser = ParserRegistry.get_for_file(rel_path)

        nodes = parser.parse_file(rel_path, source)
        if not nodes:
            return 0

        chunks = self._create_chunks(nodes, repo_id)
        if not chunks:
            return 0

        texts = [c.embedding_text for c in chunks]
        embeddings = self._embedder.encode(texts)
        self._store.add_chunks(chunks, embeddings)
        return len(chunks)

    def _create_chunks(self, nodes: list[ASTNode], repo_id: str) -> list[CodeChunk]:
        chunks = []
        for node in nodes:
            if node.node_type in CHUNKED_NODE_TYPES:
                parser = ParserRegistry.get_for_language(node.language)
                chunk_text = parser.get_chunk_text(node)
                chunk_id = self._make_chunk_id(repo_id, node)
                chunks.append(
                    CodeChunk(
                        id=chunk_id,
                        node=node,
                        embedding_text=chunk_text,
                        metadata={"repo": repo_id},
                    )
                )
        return chunks

    @staticmethod
    def _make_chunk_id(repo_id: str, node: ASTNode) -> str:
        raw = f"{repo_id}:{node.filepath}:{node.start_line}:{node.name}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
