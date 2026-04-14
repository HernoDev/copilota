"""Wrapper sobre ChromaDB para almacenamiento y búsqueda de chunks de código."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

from copilota.storage.models import CodeChunk


class VectorStore:
    """Abstrae ChromaDB para que el resto del sistema no dependa directamente de él."""

    COLLECTION_NAME = "copilota_code"

    def __init__(self, persist_directory: Path | None = None):
        if persist_directory:
            persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False),
            )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        ids = [c.id for c in chunks]
        texts = [c.embedding_text for c in chunks]
        metadatas = [c.to_chroma_metadata() for c in chunks]
        self._collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[dict]:
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        results = self._collection.query(**kwargs)
        return self._format_results(results)

    def delete_by_filepath(self, filepath: str) -> None:
        existing = self._collection.get(where={"filepath": filepath})
        if existing and existing["ids"]:
            self._collection.delete(ids=existing["ids"])

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _format_results(results: dict) -> list[dict]:
        formatted = []
        if not results.get("ids") or not results["ids"][0]:
            return formatted
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return formatted
