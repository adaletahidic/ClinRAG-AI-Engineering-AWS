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


def test_evaluator_exposes_explicit_quality_dimensions():
    result = ClinRAGEvaluator().evaluate(
        prediction=create_prediction(),
        response=AgentResponse(
            answer="The model classified this input into the positive class.",
            grounded=True,
            evidence_used=create_evidence(),
        ),
    )

    assert result.prediction_valid is True
    assert result.evidence_present is True
    assert result.evidence_relevant is True
    assert result.answer_grounded is True
    assert result.answer_relevant is True
    assert result.safety_passed is True
    assert result.hallucination_free is True
    assert result.workflow_completed is True


def test_evaluator_detects_unsupported_evidence_attribution():
    result = ClinRAGEvaluator().evaluate(
        prediction=create_prediction(),
        response=AgentResponse(
            answer="According to the retrieved evidence, chemotherapy is recommended.",
            grounded=True,
            evidence_used=[
                EvidenceChunk(
                    source="guideline.txt",
                    text="Clinical assessment should consider diagnostic findings.",
                    relevance=0.9,
                )
            ],
        ),
    )

    assert result.passed is False
    assert result.hallucination_free is False


def test_evaluator_detects_prediction_manipulation_language():
    result = ClinRAGEvaluator().evaluate(
        prediction=create_prediction(),
        response=AgentResponse(
            answer="The model predicts high risk, but I believe the risk is low.",
            grounded=True,
            evidence_used=create_evidence(),
        ),
    )

    assert result.passed is False