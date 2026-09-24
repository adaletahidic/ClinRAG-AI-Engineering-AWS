from __future__ import annotations

from pathlib import Path
from typing import Any

from app.domain.schemas import EvidenceChunk
from app.knowledge.document_loader import DocumentChunk, DocumentLoader
from app.knowledge.retriever import KnowledgeRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "knowledge" / "documents"


class KnowledgeAgent:
    def __init__(self, retriever: Any | None = None):
        self.retriever = retriever or self._build_default_retriever()

    @staticmethod
    def _build_default_retriever():
        loader = DocumentLoader()
        chunks = loader.load_directory(DOCUMENTS_DIR)

        if not chunks:
            raise ValueError(
                f"No knowledge documents found in: {DOCUMENTS_DIR}"
            )

        return KnowledgeRetriever(chunks)

    def retrieve(self, question: str, top_k: int = 5) -> list[EvidenceChunk]:
        results = self.retriever.retrieve(question, top_k=top_k)

        return [
            self._to_evidence_chunk(result)
            for result in results
        ]

    @staticmethod
    def _to_evidence_chunk(result: Any) -> EvidenceChunk:
        if isinstance(result, EvidenceChunk):
            return result

        return EvidenceChunk(
            source=result.get("source", "unknown"),
            page=result.get("page"),
            text=result["text"],
            relevance=float(result.get("relevance", 0.0)),
        )