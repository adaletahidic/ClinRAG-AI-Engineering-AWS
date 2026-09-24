from app.agents.explanation_agent import ExplanationAgent
from app.domain.schemas import (
    EvidenceChunk,
    PredictionResult,
)
from tests.fake_llm import FakeLLM


def test_explanation_agent_generates_grounded_explanation():

    prediction = PredictionResult(
        prediction=1,
        probability=0.92,
        risk_group="High",
        model_name="Bayesian-BiLSTM",
        model_version="1.0",
    )

    evidence = [
        EvidenceChunk(
            source="breast_cancer_guideline.pdf",
            page=5,
            text=(
                "Clinical assessment should consider "
                "diagnostic findings and patient history."
            ),
            relevance=0.91,
        )
    ]

    fake_llm = FakeLLM(
        response=(
            "The prediction is classified as positive with "
            "a predicted probability of 0.9200 and risk group High. "
            "The explanation is based on the supplied evidence."
        )
    )

    agent = ExplanationAgent(llm=fake_llm)

    response = agent.explain(
        prediction=prediction,
        evidence=evidence,
        question="Why was this prediction made?",
    )

    assert response.answer
    assert response.grounded is True
    assert len(response.evidence_used) == 1
    assert response.evidence_used[0].source == (
        "breast_cancer_guideline.pdf"
    )

    assert "0.9200" in response.answer
    assert "High" in response.answer

    assert len(fake_llm.calls) == 1

    assert "0.9200" in fake_llm.calls[0]["user_prompt"]
    assert "High" in fake_llm.calls[0]["user_prompt"]


def test_explanation_agent_without_evidence_is_not_grounded():

    prediction = PredictionResult(
        prediction=0,
        probability=0.12,
        risk_group="Low",
        model_name="Bayesian-BiLSTM",
        model_version="1.0",
    )

    fake_llm = FakeLLM(
        response=(
            "The model prediction is 0 with a predicted probability "
            "of 0.1200 and risk group Low. No external evidence was "
            "retrieved, so no evidence-grounded clinical explanation "
            "can be provided."
        )
    )

    agent = ExplanationAgent(llm=fake_llm)

    response = agent.explain(
        prediction=prediction,
        evidence=[],
        question="Why was this prediction made?",
    )

    assert response.answer
    assert response.grounded is False
    assert response.evidence_used == []

    assert "0.1200" in response.answer
    assert "Low" in response.answer
    assert "No external evidence" in response.answer


def test_explanation_agent_does_not_change_prediction():

    prediction = PredictionResult(
        prediction=1,
        probability=0.87,
        risk_group="High",
        model_name="Bayesian-BiLSTM",
        model_version="1.0",
    )

    fake_llm = FakeLLM(
        response=(
            "The existing prediction is explained using the "
            "provided information."
        )
    )

    agent = ExplanationAgent(llm=fake_llm)

    response = agent.explain(
        prediction=prediction,
        evidence=[],
        question="Explain the result.",
    )

    # The explanation layer does not create
    # or modify the model prediction.
    assert prediction.prediction == 1
    assert prediction.probability == 0.87
    assert prediction.risk_group == "High"

    assert response.grounded is False