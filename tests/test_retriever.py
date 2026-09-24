from app.knowledge.document_loader import DocumentChunk
from app.knowledge.retriever import KnowledgeRetriever

def test_retriever_returns_relevant_document():
    documents = [
        DocumentChunk(
            source="breast_cancer_guideline.txt",
            page=None,
            text=(
                "Breast cancer diagnosis can involve "
                "clinical examination, imaging and pathology."
            ),
        ),
        DocumentChunk(
            source="parkinson_guideline.txt",
            page=None,
            text=(
                "Parkinson disease progression can be "
                "assessed using motor symptoms and UPDRS."
            ),
        ),
        DocumentChunk(
            source="diabetes_guideline.txt",
            page=None,
            text=(
                "Diabetes management includes glucose "
                "monitoring and lifestyle interventions."
            ),
        ),
    ]

    retriever = KnowledgeRetriever(documents)

    results = retriever.retrieve(
        query="breast cancer diagnosis",
        top_k=2,
    )

    assert len(results) > 0
    assert results[0].source == "breast_cancer_guideline.txt"
    assert results[0].relevance > 0.0


def test_retriever_respects_top_k():
    documents = [
        DocumentChunk(
            source="doc1.txt",
            page=None,
            text="Breast cancer diagnosis and screening.",
        ),
        DocumentChunk(
            source="doc2.txt",
            page=None,
            text="Breast cancer treatment and pathology.",
        ),
        DocumentChunk(
            source="doc3.txt",
            page=None,
            text="Breast cancer clinical guidelines.",
        ),
    ]

    retriever = KnowledgeRetriever(documents)

    results = retriever.retrieve(
        query="breast cancer",
        top_k=2,
    )

    assert len(results) == 2


def test_retriever_rejects_empty_query():
    retriever = KnowledgeRetriever([])

    try:
        retriever.retrieve("")
        assert False
    except ValueError as exc:
        assert "Query cannot be empty" in str(exc)


def test_retriever_returns_evidence_chunks():
    documents = [
        DocumentChunk(
            source="guideline.pdf",
            page=3,
            text=(
                "Clinical assessment should consider "
                "patient history and diagnostic findings."
            ),
        )
    ]

    retriever = KnowledgeRetriever(documents)

    results = retriever.retrieve(
        query="clinical assessment",
        top_k=1,
    )

    evidence = results[0]

    assert evidence.source == "guideline.pdf"
    assert evidence.page == 3
    assert evidence.text
    assert 0.0 <= evidence.relevance <= 1.0