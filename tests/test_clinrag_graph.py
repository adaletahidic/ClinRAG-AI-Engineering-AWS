from __future__ import annotations

import pytest

from app.agents.explanation_agent import ExplanationAgent
from app.domain.schemas import EvidenceChunk, EvaluationResult
from app.evaluation.evaluator import ClinRAGEvaluator
from app.graph.clinrag_graph import ClinRAGGraph


class FakePredictionClient:
    async def predict_patient(
        self,
        features: dict[str, float],
    ) -> dict:
        return {
            "prediction": 1,
            "probability": 0.92,
            "risk_group": "High",
            "model_name": "Bayesian-BiLSTM",
            "model_version": "1.0",
        }


class FakeKnowledgeAgent:
    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[EvidenceChunk]:
        return [
            EvidenceChunk(
                source="breast_cancer_guideline.pdf",
                page=5,
                text="Clinical assessment should consider diagnostic findings.",
                relevance=0.91,
            )
        ]


class FailingPredictionClient:
    async def predict_patient(
        self,
        features: dict[str, float],
    ) -> dict:
        raise RuntimeError("MCP server unavailable")


class FailingEvaluator:
    def evaluate(
        self,
        prediction,
        response,
        require_evidence=True,
    ) -> EvaluationResult:
        return EvaluationResult(
            passed=False,
            grounded=False,
            safe=False,
            relevant=False,
            issues=["Evaluation failed intentionally."],
        )


@pytest.mark.asyncio
async def test_clinrag_graph_runs_end_to_end():
    graph = ClinRAGGraph(
        prediction_client=FakePredictionClient(),
        knowledge_agent=FakeKnowledgeAgent(),
        explanation_agent=ExplanationAgent(),
        evaluator=ClinRAGEvaluator(),
    )

    result = await graph.run(
        question="Explain the model prediction using the available evidence.",
        features={"feature_1": 1.0},
    )

    assert result["prediction"]["prediction"] == 1
    assert result["prediction"]["probability"] == 0.92
    assert result["prediction"]["risk_group"] == "High"

    assert len(result["evidence"]) == 1

    assert result["answer"]

    assert result["grounded"] is True

    assert result["evaluation_passed"] is True
    assert result["safe"] is True
    assert result["relevant"] is True

    assert result["evaluation_issues"] == []
    assert result["errors"] == []

    assert result["safe_failure"] is False
    assert any(entry["step"] == "workflow_start" for entry in result["trace"])
    assert any(entry["step"] == "prediction" for entry in result["trace"])
    assert any(entry["step"] == "knowledge" for entry in result["trace"])
    assert any(entry["step"] == "explanation" for entry in result["trace"])
    assert any(entry["step"] == "evaluation" for entry in result["trace"])


@pytest.mark.asyncio
async def test_clinrag_graph_uses_safe_failure_when_prediction_fails():
    graph = ClinRAGGraph(
        prediction_client=FailingPredictionClient(),
        knowledge_agent=FakeKnowledgeAgent(),
        explanation_agent=ExplanationAgent(),
        evaluator=ClinRAGEvaluator(),
    )

    result = await graph.run(
        question="Explain the model prediction.",
        features={"feature_1": 1.0},
    )

    assert result["safe_failure"] is True

    assert "prediction" not in result

    assert result["grounded"] is False

    assert result["answer"]

    assert len(result["errors"]) == 1
    assert "Prediction step failed" in result["errors"][0]

    assert "MCP server unavailable" in result["errors"][0]


@pytest.mark.asyncio
async def test_clinrag_graph_uses_safe_failure_when_evaluation_fails():
    graph = ClinRAGGraph(
        prediction_client=FakePredictionClient(),
        knowledge_agent=FakeKnowledgeAgent(),
        explanation_agent=ExplanationAgent(),
        evaluator=FailingEvaluator(),
    )

    result = await graph.run(
        question="Explain the model prediction.",
        features={"feature_1": 1.0},
    )

    assert result["safe_failure"] is True

    assert result["evaluation_passed"] is False

    assert result["safe"] is False
    assert result["grounded"] is False
    assert result["relevant"] is False

    assert result["answer"]

    assert "Evaluation failed intentionally." in result["evaluation_issues"]