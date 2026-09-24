from __future__ import annotations

import pytest

from app.agents.explanation_agent import ExplanationAgent
from app.evaluation.evaluator import ClinRAGEvaluator
from app.graph.clinrag_graph import ClinRAGGraph
from tests.test_clinrag_graph import FakeKnowledgeAgent, FakePredictionClient


@pytest.mark.asyncio
async def test_graph_emits_request_id_timestamps_and_latency():
    graph = ClinRAGGraph(
        prediction_client=FakePredictionClient(),
        knowledge_agent=FakeKnowledgeAgent(),
        explanation_agent=ExplanationAgent(),
        evaluator=ClinRAGEvaluator(),
    )

    result = await graph.run(
        question="Explain the model prediction.",
        features={"feature_1": 1.0},
        request_id="request-123",
    )

    assert result["request_id"] == "request-123"
    assert result["workflow_latency_ms"] >= 0
    assert result["trace"]
    assert all(event["request_id"] == "request-123" for event in result["trace"])
    assert all("timestamp" in event for event in result["trace"])
    assert all("latency_ms" in event for event in result["trace"][1:])
