from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """
    Aggregate metrics for ClinRAG evaluation runs.
    """

    total_cases: int
    passed_cases: int
    failed_cases: int
    safety_failures: int = 0
    grounding_failures: int = 0
    retrieval_failures: int = 0

    @property
    def pass_rate(self) -> float:
        if self.total_cases == 0:
            return 0.0

        return self.passed_cases / self.total_cases

    @property
    def failure_rate(self) -> float:
        if self.total_cases == 0:
            return 0.0

        return self.failed_cases / self.total_cases


def calculate_metrics(
    passed_cases: int,
    failed_cases: int,
    safety_failures: int = 0,
    grounding_failures: int = 0,
    retrieval_failures: int = 0,
) -> EvaluationMetrics:

    total_cases = (
        passed_cases + failed_cases
    )

    return EvaluationMetrics(
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        safety_failures=safety_failures,
        grounding_failures=grounding_failures,
        retrieval_failures=retrieval_failures,
    )