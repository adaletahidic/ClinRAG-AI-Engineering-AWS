from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from app.evaluation.metrics import EvaluationMetrics, calculate_metrics
from app.evaluation.test_cases import (
    DEFAULT_TEST_CASES,
    EvaluationTestCase,
)


Workflow = Callable[
    [str, dict[str, float]],
    Awaitable[dict],
]


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    passed: bool
    expected_passed: bool
    safe_failure: bool
    issues: list[str]


@dataclass(frozen=True)
class EvaluationRun:
    results: list[ScenarioResult]
    metrics: EvaluationMetrics


class EvaluationScenarioRunner:
    """Runs repeatable workflow scenarios without calling provider APIs itself."""

    def __init__(
        self,
        workflow: Workflow,
        features: dict[str, float] | None = None,
    ):
        self.workflow = workflow
        self.features = features or {}

    async def run(
        self,
        cases: list[EvaluationTestCase] | None = None,
    ) -> EvaluationRun:
        scenarios = cases or DEFAULT_TEST_CASES
        results: list[ScenarioResult] = []

        for case in scenarios:
            state = await self.workflow(
                case.question,
                case.features if case.features is not None else self.features,
            )
            passed = bool(state.get("evaluation_passed", False))
            safe_failure = bool(state.get("safe_failure", False))
            issues = list(state.get("evaluation_issues", []))
            issues.extend(state.get("errors", []))

            scenario_passed = (
                passed == case.expected_passed
                and safe_failure == case.expected_safe_failure
            )

            results.append(
                ScenarioResult(
                    name=case.name,
                    passed=scenario_passed,
                    expected_passed=case.expected_passed,
                    safe_failure=safe_failure,
                    issues=issues,
                )
            )

        passed_cases = sum(item.passed for item in results)
        failed_cases = len(results) - passed_cases
        return EvaluationRun(
            results=results,
            metrics=calculate_metrics(
                passed_cases=passed_cases,
                failed_cases=failed_cases,
            ),
        )
