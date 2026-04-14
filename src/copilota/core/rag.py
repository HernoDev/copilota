"""Pipeline RAG: combina retriever + LLM para responder preguntas sobre código."""

from __future__ import annotations

from copilota.core.retriever import RetrievalResult, Retriever
from copilota.llm.base import BaseLLM


DEFAULT_SYSTEM_PROMPT = (
    "Eres un asistente experto en código. "
    "Responde preguntas basándote en los fragmentos de código proporcionados. "
    "Cita archivos y líneas cuando sea posible."
)


class RAGPipeline:
    """Orquesta retrieval + generación con LLM."""

    def __init__(self, retriever: Retriever, llm: BaseLLM):
        self._retriever = retriever
        self._llm = llm

    async def query(
        self,
        question: str,
        top_k: int = 5,
        language: str | None = None,
    ) -> dict:
        results = self._retriever.search(question, top_k=top_k, language=language)

        context = self._build_context(results)
        prompt = self._build_prompt(question, context)

        answer = await self._llm.generate(
            prompt=prompt,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )

        return {
            "answer": answer,
            "sources": [
                {
                    "filepath": r.filepath,
                    "name": r.name,
                    "node_type": r.node_type,
                    "score": round(r.score, 3),
                }
                for r in results
            ],
        }

    @staticmethod
    def _build_context(results: list[RetrievalResult]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"--- Fragment {i} ---")
            parts.append(f"File: {r.filepath}")
            parts.append(f"Type: {r.node_type} | Name: {r.name}")
            parts.append(r.document)
            parts.append("")
        return "\n".join(parts) if parts else "No se encontraron fragmentos relevantes."

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        return (
            f"Contexto del código:\n{context}\n\n"
            f"Pregunta: {question}\n\n"
            f"Responde la pregunta usando el contexto proporcionado."
        )
