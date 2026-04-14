"""Generador de embeddings: abstrae sentence-transformers para uso local o mock."""

from __future__ import annotations


class EmbeddingModel:
    """Genera embeddings de texto. Usa sentence-transformers si está disponible, sino mock."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_mock: bool = False):
        self._model_name = model_name
        self._use_mock = use_mock or self._check_mock_needed()
        self._model = None if self._use_mock else self._load_model()

    def _check_mock_needed(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
            return False
        except ImportError:
            return True

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self._model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._use_mock or self._model is None:
            return self._mock_encode(texts)
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def encode_single(self, text: str) -> list[float]:
        result = self.encode([text])
        return result[0]

    def _mock_encode(self, texts: list[str]) -> list[list[float]]:
        import hashlib
        results = []
        for text in texts:
            h = hashlib.md5(text.encode()).hexdigest()
            vec = [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]
            vec = vec + [0.0] * (384 - len(vec))
            results.append(vec)
        return results
