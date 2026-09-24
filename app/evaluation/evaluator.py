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
        workflow_completed: bool = True,
    ) -> EvaluationResult:

        issues: list[str] = []

        prediction_valid = self._validate_prediction(
            prediction,
            issues,
        )
        prediction_integrity = self._check_prediction_integrity(
            prediction,
            response,
            issues,
        )

        response_valid = self._validate_response(
            response,
            issues,
        )

        evidence_present = bool(response and response.evidence_used)
        evidence_relevant = self._check_evidence_relevance(
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

        hallucination_free = self._check_hallucination_risk(
            response,
            issues,
        )

        if not workflow_completed:
            issues.append("Workflow did not complete successfully.")

        passed = (
            prediction_valid
            and prediction_integrity
            and response_valid
            and grounded
            and safe
            and relevant
            and evidence_relevant
            and hallucination_free
            and workflow_completed
            and not issues
        )

        return EvaluationResult(
            passed=passed,
            grounded=grounded,
            safe=safe,
            relevant=relevant,
            issues=issues,
            prediction_valid=prediction_valid,
            prediction_integrity=prediction_integrity,
            evidence_present=evidence_present,
            evidence_relevant=evidence_relevant,
            answer_grounded=grounded,
            answer_relevant=relevant,
            safety_passed=safe,
            hallucination_free=hallucination_free,
            workflow_completed=workflow_completed,
        )

    @staticmethod
    def _check_evidence_relevance(
        response: AgentResponse | None,
        issues: list[str],
    ) -> bool:
        if response is None or not response.evidence_used:
            return False

        if not any(item.relevance > 0 for item in response.evidence_used):
            issues.append("Retrieved evidence has no positive relevance score.")
            return False

        return True

    @staticmethod
    def _check_prediction_integrity(
        prediction: PredictionResult | None,
        response: AgentResponse | None,
        issues: list[str],
    ) -> bool:
        if prediction is None or response is None:
            return False

        answer = response.answer.lower()
        opposite_group = "low" if prediction.risk_group == "High" else "high"
        if (
            "but i believe" in answer
            and opposite_group in answer
        ):
            issues.append(
                "Response attempts to reinterpret the authoritative "
                "prediction."
            )
            return False

        for value in ("0", "1"):
            if (
                f"prediction is {value}" in answer
                and int(value) != prediction.prediction
            ):
                issues.append(
                    "Response contradicts the authoritative prediction value."
                )
                return False

        return True

    @staticmethod
    def _check_hallucination_risk(
        response: AgentResponse | None,
        issues: list[str],
    ) -> bool:
        if response is None:
            return False

        answer = response.answer.lower()
        unsupported_claim_markers = (
            "the evidence recommends",
            "the guideline recommends",
            "according to the retrieved evidence",
            "according to the guideline",
        )

        if any(marker in answer for marker in unsupported_claim_markers):
            evidence_text = " ".join(
                item.text.lower() for item in response.evidence_used
            )
            ignored_tokens = {
                "a", "an", "and", "according", "by", "evidence", "for",
                "from", "is", "it", "of", "retrieved", "the", "to",
            }
            claim_tokens = {
                token.strip(".,:;!?()[]")
                for token in answer.split()
                if token.strip(".,:;!?()[]") not in ignored_tokens
            }
            evidence_tokens = {
                token.strip(".,:;!?()[]")
                for token in evidence_text.split()
                if token.strip(".,:;!?()[]") not in ignored_tokens
            }

            if not claim_tokens.intersection(evidence_tokens):
                issues.append(
                    "Response contains an evidence attribution unsupported "
                    "by the retrieved text."
                )
                return False

        return True

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