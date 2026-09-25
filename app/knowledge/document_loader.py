from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentChunk:
    """
    A chunk of source documentation used by the RAG layer.
    """

    source: str
    page: int | None
    text: str


class DocumentLoader:
    """
    Loads source documents and splits them into chunks.

    The loader does not perform retrieval or generation.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(
        self,
        path: str | Path,
    ) -> list[DocumentChunk]:
        """
        Load a PDF or text document and return chunks.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(path)

        if suffix in {".txt", ".md"}:
            return self._load_text(path)

        raise ValueError(
            f"Unsupported document type: {suffix}"
        )

    def load_directory(
        self,
        directory: str | Path,
    ) -> list[DocumentChunk]:
        """
        Load all supported documents from a directory.
        """

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        chunks: list[DocumentChunk] = []

        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in {
                ".pdf",
                ".txt",
                ".md",
            }:
                continue

            chunks.extend(self.load(path))

        return chunks

    def _load_pdf(
        self,
        path: Path,
    ) -> list[DocumentChunk]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "pypdf is required for PDF loading. "
                "Install it with: "
                "python -m pip install pypdf"
            ) from exc

        reader = PdfReader(str(path))

        chunks: list[DocumentChunk] = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            text = page.extract_text() or ""

            text = self._normalize_text(text)

            if not text:
                continue

            page_chunks = self._chunk_text(text)

            for chunk in page_chunks:
                chunks.append(
                    DocumentChunk(
                        source=str(path),
                        page=page_number,
                        text=chunk,
                    )
                )

        return chunks

    def _load_text(
        self,
        path: Path,
    ) -> list[DocumentChunk]:
        text = path.read_text(
            encoding="utf-8"
        )

        text = self._normalize_text(text)

        if not text:
            return []

        return self._load_text_from_content(
            text,
            source=str(path),
        )

    def _load_text_from_content(
        self,
        text: str,
        source: str,
    ) -> list[DocumentChunk]:
        text = self._normalize_text(text)

        if not text:
            return []

        return [
            DocumentChunk(
                source=source,
                page=None,
                text=chunk,
            )
            for chunk in self._chunk_text(text)
        ]

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        lines = [
            line.strip()
            for line in text.splitlines()
        ]

        return "\n".join(
            line
            for line in lines
            if line
        )

    def _chunk_text(
        self,
        text: str,
    ) -> list[str]:
        """
        Split text using character-based overlapping windows.

        This is deliberately simple for the first RAG iteration.
        """

        chunks: list[str] = []

        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(
                start + self.chunk_size,
                text_length,
            )

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = end - self.chunk_overlap

        return chunks