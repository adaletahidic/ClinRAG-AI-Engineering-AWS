from __future__ import annotations

from typing import Protocol

from app.domain.schemas import EvidenceChunk
from app.knowledge.document_loader import DocumentChunk


class KnowledgeSource(Protocol):
    def load_documents(self) -> list[DocumentChunk]:
        ...


class VectorStore(Protocol):
    def add_documents(self, documents: list[DocumentChunk]) -> None:
        ...

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[EvidenceChunk]:
        ...
