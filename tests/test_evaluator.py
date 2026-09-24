from app.domain.schemas import (
    AgentResponse,
    EvidenceChunk,
    PredictionResult,
)
from app.evaluation.evaluator import ClinRAGEvaluator


def create_prediction() -> PredictionResult:
    return PredictionResult(
        prediction=1,
        probability=0.92,
        risk_group="High",
        model_name="Bayesian-BiLSTM",
        model_version="1.0",
    )


def create_evidence() -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            source="breast_cancer_guideline.pdf",
            page=5,
            text=(
                "Clinical assessment should consider "
                "diagnostic findings."
            ),
            relevance=0.91,
        )
    ]


def test_evaluator_passes_valid_grounded_response():

    prediction = create_prediction()

    response = AgentResponse(
        answer=(
            "The model classified this input into "
            "the positive prediction class."
        ),
        grounded=True,
        evidence_used=create_evidence(),
    )

    evaluator = ClinRAGEvaluator()

    result = evaluator.evaluate(
        prediction=prediction,
        response=response,
    )

    assert result.passed is True
    assert result.grounded is True
    assert result.safe is True
    assert result.relevant is True
    assert result.issues == []


def test_evaluator_fails_without_evidence():

    prediction = create_prediction()

    response = AgentResponse(
        answer="The model classified this input.",
        grounded=False,
        evidence_used=[],
    )

    evaluator = ClinRAGEvaluator()

    result = evaluator.evaluate(
        prediction=prediction,
        response=response,
    )

    assert result.passed is False
    assert result.grounded is False
    assert result.safe is True
    assert result.relevant is True
    assert result.issues


def test_evaluator_blocks_unsafe_diagnostic_language():

    prediction = create_prediction()

    response = AgentResponse(
        answer="You definitely have cancer.",
        grounded=True,
        evidence_used=create_evidence(),
    )

    evaluator = ClinRAGEvaluator()

    result = evaluator.evaluate(
        prediction=prediction,
        response=response,
    )

    assert result.passed is False
    assert result.safe is False
    assert any(
        "unsafe diagnostic" in issue.lower()
        for issue in result.issues
    )


def test_evaluator_fails_missing_prediction():

    response = AgentResponse(
        answer="The model generated a result.",
        grounded=True,
        evidence_used=create_evidence(),
    )

    evaluator = ClinRAGEvaluator()

    result = evaluator.evaluate(
        prediction=None,
        response=response,
    )

    assert result.passed is False
    assert result.safe is True
    assert result.issues