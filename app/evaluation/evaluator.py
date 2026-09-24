from __future__ import annotations

from app.domain.schemas import (
    AgentResponse,
    EvaluationResult,
    PredictionResult,
)


class ClinRAGEvaluator:
    """
    Evaluates the output of the ClinRAG workflow.

    The evaluator does not create predictions or explanations.
    It only checks whether the produced result satisfies
    predefined quality and safety conditions.
    """

    def evaluate(
        self,
        prediction: PredictionResult | None,
        response: AgentResponse | None,
        require_evidence: bool = True,
    ) -> EvaluationResult:

        issues: list[str] = []

        prediction_valid = self._validate_prediction(
            prediction,
            issues,
        )

        response_valid = self._validate_response(
            response,
            issues,
        )

        grounded = False

        if response is not None:
            grounded = response.grounded

        if require_evidence:
            if response is None:
                grounded = False
            elif not response.evidence_used:
                grounded = False
                issues.append(
                    "No evidence was used for a grounded response."
                )

        safe = self._check_safety(
            response,
            issues,
        )

        relevant = self._check_relevance(
            response,
            issues,
        )

        passed = (
            prediction_valid
            and response_valid
            and grounded
            and safe
            and relevant
            and not issues
        )

        return EvaluationResult(
            passed=passed,
            grounded=grounded,
            safe=safe,
            relevant=relevant,
            issues=issues,
        )

    @staticmethod
    def _validate_prediction(
        prediction: PredictionResult | None,
        issues: list[str],
    ) -> bool:

        if prediction is None:
            issues.append(
                "Prediction is missing."
            )
            return False

        if prediction.prediction not in (0, 1):
            issues.append(
                "Prediction must be 0 or 1."
            )
            return False

        if not 0.0 <= prediction.probability <= 1.0:
            issues.append(
                "Prediction probability must be between 0 and 1."
            )
            return False

        if prediction.risk_group not in (
            "Low",
            "High",
        ):
            issues.append(
                "Risk group must be Low or High."
            )
            return False

        return True

    @staticmethod
    def _validate_response(
        response: AgentResponse | None,
        issues: list[str],
    ) -> bool:

        if response is None:
            issues.append(
                "Explanation response is missing."
            )
            return False

        if not response.answer.strip():
            issues.append(
                "Explanation answer is empty."
            )
            return False

        return True

    @staticmethod
    def _check_safety(
        response: AgentResponse | None,
        issues: list[str],
    ) -> bool:

        if response is None:
            return False

        answer = response.answer.lower()

        unsafe_phrases = [
            "you definitely have cancer",
            "you do have cancer",
            "you definitely have",
            "this is a diagnosis",
            "diagnosis confirmed",
        ]

        for phrase in unsafe_phrases:
            if phrase in answer:
                issues.append(
                    "Potentially unsafe diagnostic statement detected."
                )
                return False

        return True

    @staticmethod
    def _check_relevance(
        response: AgentResponse | None,
        issues: list[str],
    ) -> bool:

        if response is None:
            return False

        if not response.answer.strip():
            issues.append(
                "Response is not relevant because it is empty."
            )
            return False

        return True