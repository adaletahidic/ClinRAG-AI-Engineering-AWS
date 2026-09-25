from pathlib import Path

import pytest

from app.knowledge.document_loader import DocumentLoader


def test_document_loader_loads_text_file(tmp_path: Path):
    document = tmp_path / "guideline.txt"

    document.write_text(
        "Breast cancer screening should be based "
        "on clinical assessment and appropriate guidelines.",
        encoding="utf-8",
    )

    loader = DocumentLoader(
        chunk_size=1000,
        chunk_overlap=100,
    )

    chunks = loader.load(document)

    assert len(chunks) == 1
    assert chunks[0].source == str(document)
    assert chunks[0].page is None
    assert "Breast cancer screening" in chunks[0].text


def test_document_loader_splits_large_text(tmp_path: Path):
    document = tmp_path / "large.txt"

    document.write_text(
        "Breast cancer clinical guidance. " * 100,
        encoding="utf-8",
    )

    loader = DocumentLoader(
        chunk_size=200,
        chunk_overlap=50,
    )

    chunks = loader.load(document)

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.text


def test_document_loader_rejects_unsupported_file(
    tmp_path: Path,
):
    document = tmp_path / "data.csv"

    document.write_text(
        "a,b,c\n1,2,3",
        encoding="utf-8",
    )

    loader = DocumentLoader()

    with pytest.raises(ValueError):
        loader.load(document)


def test_document_loader_missing_file():
    loader = DocumentLoader()

    with pytest.raises(FileNotFoundError):
        loader.load("does_not_exist.pdf")


def test_document_loader_reads_uploaded_text_bytes():
    loader = DocumentLoader()

    chunks = loader.load_bytes(
        b"Breast screening evidence and risk stratification.",
        "uploaded-guidance.txt",
    )

    assert len(chunks) == 1
    assert chunks[0].source == "uploaded-guidance.txt"
    assert "risk stratification" in chunks[0].text