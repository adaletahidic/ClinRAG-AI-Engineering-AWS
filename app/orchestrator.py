from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ExecutionStatus = Literal[
    "initialized",
    "prediction_completed",
    "retrieval_completed",
    "explanation_completed",
    "evaluation_completed",
    "failed",
]


@dataclass
class PredictionContext:
    """
    Structured output produced by the predictive model.

    The LLM must never become the authority for these fields.
    """

    prediction: int
    probability: float
    risk_group: str
    model_name: str
    model_version: str


@dataclass
class EvidenceContext:
    """
    Evidence retrieved from the clinical knowledge base.
    """

    source: str
    page: int | None
    text: str
    relevance: float


@dataclass
class EvaluationResult:
    """
    Result of evaluating the generated response.
    """

    passed: bool
    grounded: bool
    safe: bool
    relevant: bool
    issues: list[str] = field(default_factory=list)


@dataclass
class OrchestrationState:
    """
    State shared between the different stages of the workflow.

    This is deliberately explicit.

    We do not allow agents to implicitly pass arbitrary information
    through prompts or global state.
    """

    user_question: str

    patient_features: dict[str, float] | None = None

    prediction: PredictionContext | None = None

    evidence: list[EvidenceContext] = field(default_factory=list)

    generated_answer: str | None = None

    evaluation: EvaluationResult | None = None

    status: ExecutionStatus = "initialized"

    trace_id: str | None = None

    errors: list[str] = field(default_factory=list)


class ClinRAGOrchestrator:
    """
    Main workflow coordinator.

    Important architectural rule:

        Prediction model
                ↓
        structured prediction
                ↓
        evidence retrieval
                ↓
        explanation
                ↓
        evaluation

    The LLM is NOT responsible for generating or modifying
    the prediction.
    """

    def __init__(
        self,
        prediction_agent: Any,
        knowledge_agent: Any,
        explanation_agent: Any,
        evaluator: Any,
    ) -> None:

        self.prediction_agent = prediction_agent
        self.knowledge_agent = knowledge_agent
        self.explanation_agent = explanation_agent
        self.evaluator = evaluator

    def run(
        self,
        *,
        user_question: str,
        patient_features: dict[str, float] | None = None,
        trace_id: str | None = None,
    ) -> OrchestrationState:

        state = OrchestrationState(
            user_question=user_question,
            patient_features=patient_features,
            trace_id=trace_id,
        )

        try:

            # ---------------------------------------------------------
            # STEP 1 — Prediction
            # ---------------------------------------------------------

            if patient_features is not None:

                prediction = self.prediction_agent.predict(
                    patient_features
                )

                state.prediction = PredictionContext(
                    prediction=prediction["prediction"],
                    probability=prediction["probability"],
                    risk_group=prediction["risk_group"],
                    model_name=prediction["model_name"],
                    model_version=prediction["model_version"],
                )

                state.status = "prediction_completed"

            # ---------------------------------------------------------
            # STEP 2 — Knowledge retrieval
            # ---------------------------------------------------------

            evidence = self.knowledge_agent.retrieve(
                query=user_question
            )

            state.evidence = [
                EvidenceContext(
                    source=item["source"],
                    page=item.get("page"),
                    text=item["text"],
                    relevance=item["relevance"],
                )
                for item in evidence
            ]

            state.status = "retrieval_completed"

            # ---------------------------------------------------------
            # STEP 3 — Explanation
            # ---------------------------------------------------------

            state.generated_answer = (
                self.explanation_agent.generate(
                    question=user_question,
                    prediction=state.prediction,
                    evidence=state.evidence,
                )
            )

            state.status = "explanation_completed"

            # ---------------------------------------------------------
            # STEP 4 — Evaluation
            # ---------------------------------------------------------

            evaluation = self.evaluator.evaluate(
                question=user_question,
                prediction=state.prediction,
                evidence=state.evidence,
                answer=state.generated_answer,
            )

            state.evaluation = EvaluationResult(
                passed=evaluation["passed"],
                grounded=evaluation["grounded"],
                safe=evaluation["safe"],
                relevant=evaluation["relevant"],
                issues=evaluation.get("issues", []),
            )

            state.status = "evaluation_completed"

            # ---------------------------------------------------------
            # STEP 5 — Safety gate
            # ---------------------------------------------------------

            if not state.evaluation.passed:

                state.generated_answer = (
                    "The generated response did not pass the "
                    "AI safety and grounding checks. "
                    "No unsupported clinical conclusion is provided."
                )

            return state

        except Exception as exc:

            state.status = "failed"
            state.errors.append(str(exc))

            state.generated_answer = (
                "The AI workflow could not complete safely. "
                "No clinical conclusion is provided."
            )

            return state