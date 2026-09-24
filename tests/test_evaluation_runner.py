from __future__ import annotations

import pytest

from app.evaluation.runner import EvaluationScenarioRunner
from app.evaluation.test_cases import EvaluationTestCase


@pytest.mark.asyncio
async def test_evaluation_runner_aggregates_expected_scenarios():
    async def workflow(question: str, features: dict[str, float]) -> dict:
        if not features:
            return {
                "evaluation_passed": False,
                "safe_failure": True,
                "errors": ["missing feature"],
            }
        return {
            "evaluation_passed": True,
            "safe_failure": False,
            "evaluation_issues": [],
            "errors": [],
        }

    run = await EvaluationScenarioRunner(
        workflow,
        features={"feature_1": 1.0},
    ).run(
        [
            EvaluationTestCase(
                name="normal",
                question="Explain.",
            ),
            EvaluationTestCase(
                name="missing_feature",
                question="Explain.",
                features={},
                expected_passed=False,
                expected_safe_failure=True,
            ),
        ]
    )

    assert run.metrics.total_cases == 2
    assert run.metrics.passed_cases == 2
    assert all(result.passed for result in run.results)
