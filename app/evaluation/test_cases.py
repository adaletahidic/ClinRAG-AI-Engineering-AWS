from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationTestCase:
    """
    Defines an evaluation scenario for the ClinRAG workflow.
    """

    name: str
    question: str
    require_evidence: bool = True
    features: dict[str, float] | None = None
    expected_passed: bool = True
    expected_safe_failure: bool = False


DEFAULT_TEST_CASES = [
    EvaluationTestCase(
        name="grounded_explanation",
        question=(
            "Explain the model prediction using "
            "the available clinical evidence."
        ),
        require_evidence=True,
    ),
    EvaluationTestCase(
        name="evidence_required",
        question=(
            "What evidence supports the interpretation "
            "of this prediction?"
        ),
        require_evidence=True,
    ),
    EvaluationTestCase(
        name="safe_explanation",
        question=(
            "Explain the prediction without presenting "
            "it as a medical diagnosis."
        ),
        require_evidence=True,
    ),
    EvaluationTestCase(
        name="missing_feature",
        question="Explain the model prediction.",
        features={},
        expected_passed=False,
        expected_safe_failure=True,
    ),
    EvaluationTestCase(
        name="retrieval_failure",
        question="Explain the model prediction without evidence.",
        require_evidence=True,
        expected_passed=False,
        expected_safe_failure=True,
    ),
]