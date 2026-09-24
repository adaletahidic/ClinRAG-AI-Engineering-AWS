from app.domain.schemas import (
    PatientFeatures,
    PredictionResult,
    EvidenceChunk,
    EvaluationResult,
)


def test_patient_features():
    patient = PatientFeatures(
        features={
            "radius_mean": 14.0,
            "texture_mean": 20.0,
        }
    )

    assert patient.features["radius_mean"] == 14.0


def test_prediction_result():
    result = PredictionResult(
        prediction=1,
        probability=0.98,
        risk_group="High",
        model_name="Bayesian-BiLSTM",
        model_version="1.0",
    )

    assert result.prediction == 1
    assert 0.0 <= result.probability <= 1.0


def test_evidence_chunk():
    evidence = EvidenceChunk(
        source="guideline.pdf",
        page=12,
        text="Example clinical guideline text.",
        relevance=0.91,
    )

    assert evidence.source == "guideline.pdf"
    assert evidence.relevance == 0.91


def test_evaluation_result():
    result = EvaluationResult(
        passed=True,
        grounded=True,
        safe=True,
        relevant=True,
    )

    assert result.passed is True