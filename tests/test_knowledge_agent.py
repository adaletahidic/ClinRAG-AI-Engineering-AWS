from __future__ import annotations

from pathlib import Path

from app.agents.knowledge_agent import KnowledgeAgent
from app.knowledge.document_loader import DocumentChunk, DocumentLoader
from app.knowledge.retriever import KnowledgeRetriever


def test_knowledge_agent_retrieves_evidence():
    documents = [
        DocumentChunk(
            source="breast_cancer_guideline.pdf",
            page=5,
            text=(
                "Breast cancer diagnosis may involve "
                "clinical examination and diagnostic imaging."
            ),
        ),
        DocumentChunk(
            source="other_guideline.pdf",
            page=2,
            text=(
                "Parkinson disease progression can be "
                "evaluated using motor symptoms."
            ),
        ),
    ]

    retriever = KnowledgeRetriever(documents)
    agent = KnowledgeAgent(retriever)

    evidence = agent.retrieve(
        question="How is breast cancer diagnosis assessed?",
        top_k=1,
    )

    assert len(evidence) == 1
    assert evidence[0].source == "breast_cancer_guideline.pdf"
    assert evidence[0].page == 5
    assert evidence[0].relevance > 0.0


def test_knowledge_agent_retrieves_from_real_document():
    project_root = Path(__file__).resolve().parents[1]
    documents_dir = project_root / "app" / "knowledge" / "documents"

    loader = DocumentLoader()
    chunks = loader.load_directory(documents_dir)

    assert chunks, "No knowledge documents were loaded."

    retriever = KnowledgeRetriever(chunks)
    agent = KnowledgeAgent(retriever=retriever)

    evidence = agent.retrieve(
        "Can the model prediction be considered a medical diagnosis?",
        top_k=3,
    )

    assert evidence

    assert any(
        "medical diagnosis" in item.text.lower()
        for item in evidence
    )


def test_default_knowledge_agent_loads_documents():
    agent = KnowledgeAgent()

    evidence = agent.retrieve(
        "What is the role of human oversight in the AI system?",
        top_k=3,
    )

    assert evidence

    assert any(
        "human" in item.text.lower()
        for item in evidence
    )