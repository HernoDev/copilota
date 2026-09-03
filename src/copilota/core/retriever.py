"""Retriever: busca chunks relevantes en la vector DB dado una query."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import git

from copilota.core.embedder import EmbeddingModel
from copilota.storage.vector_db import VectorStore


def resolve_repo(repo: str) -> str:
    """Resuelve cualquier ruta dentro de un repo a su raíz (working tree)."""
    p = Path(repo).resolve()
    try:
        return str(Path(git.Repo(str(p), search_parent_directories=True).working_tree_dir))
    except Exception:
        return str(p)


@dataclass
class RetrievalResult:
    id: str
    document: str
    filepath: str
    language: str
    node_type: str
    name: str
    score: float


class Retriever:
    """Busca código relevante usando embeddings."""

    def __init__(self, vector_store: VectorStore, embedder: EmbeddingModel):
        self._store = vector_store
        self._embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 5,
        language: str | None = None,
        repo: str | None = None,
    ) -> list[RetrievalResult]:
        query_embedding = self._embedder.encode_single(query)

        filters: dict[str, str] = {}
        if language:
            filters["language"] = language
        if repo:
            filters["repo"] = resolve_repo(repo)

        raw_results = self._store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        results = []
        for r in raw_results:
            meta = r["metadata"]
            results.append(
                RetrievalResult(
                    id=r["id"],
                    document=r["document"],
                    filepath=meta.get("filepath", ""),
                    language=meta.get("language", ""),
                    node_type=meta.get("node_type", ""),
                    name=meta.get("name", ""),
                    score=1.0 - r["distance"],
                )
            )
        return results
