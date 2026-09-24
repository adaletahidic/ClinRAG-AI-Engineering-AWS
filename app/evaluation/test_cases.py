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
]