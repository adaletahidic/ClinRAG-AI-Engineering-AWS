from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.domain.schemas import EvidenceChunk
from app.knowledge.document_loader import DocumentChunk

@dataclass
class RetrievalResult:
    """
    Internal representation of a retrieved document.
    """

    source: str
    page: int | None
    text: str
    relevance: float


class KnowledgeRetriever:
    """
    Local lexical retriever based on TF-IDF and cosine similarity.

    Responsibilities:
    - index document chunks
    - retrieve relevant chunks
    - return normalized EvidenceChunk objects

    It does not generate text and does not modify predictions.
    """

    def __init__(
        self,
        documents: list[DocumentChunk] | None = None,
    ):
        self.documents = documents or []

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
        )

        self.document_matrix = None

        if self.documents:
            self._build_index()

    def add_documents(
        self,
        documents: list[DocumentChunk],
    ) -> None:
        self.documents.extend(documents)

        self._build_index()

    def _build_index(self) -> None:
        if not self.documents:
            self.document_matrix = None
            return

        texts = [
            document.text
            for document in self.documents
        ]

        self.document_matrix = (
            self.vectorizer.fit_transform(texts)
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[EvidenceChunk]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if not self.documents:
            return []

        if self.document_matrix is None:
            self._build_index()

        query_vector = (
            self.vectorizer.transform([query])
        )

        scores = cosine_similarity(
            query_vector,
            self.document_matrix,
        )[0]

        ranked_indices = scores.argsort()[::-1]

        results: list[EvidenceChunk] = []

        for index in ranked_indices[:top_k]:
            score = float(scores[index])

            # Ignore completely unrelated documents.
            if score <= 0:
                continue

            document = self.documents[index]

            results.append(
                EvidenceChunk(
                    source=document.source,
                    page=document.page,
                    text=document.text,
                    relevance=score,
                )
            )

        return results

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:

        evidence = self.retrieve(
            query=query,
            top_k=top_k,
        )

        return [
            RetrievalResult(
                source=item.source,
                page=item.page,
                text=item.text,
                relevance=item.relevance,
            )
            for item in evidence
        ]