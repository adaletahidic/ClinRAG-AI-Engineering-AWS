from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.explanation_agent import ExplanationAgent
from app.agents.knowledge_agent import KnowledgeAgent
from app.domain.schemas import (
    AgentResponse,
    EvidenceChunk,
    PredictionResult,
)
from app.evaluation.evaluator import ClinRAGEvaluator
from app.mcp.prediction_client import PredictionMCPClient


class ClinRAGState(TypedDict, total=False):
    question: str
    features: dict[str, float]

    prediction: dict
    evidence: list[dict]

    answer: str
    grounded: bool

    evaluation_passed: bool
    safe: bool
    relevant: bool
    evaluation_issues: list[str]

    errors: list[str]
    safe_failure: bool


class ClinRAGGraph:
    """
    LangGraph orchestration layer for ClinRAG.

    Flow:

        START
          |
      prediction
          |
      success?
       /    \
     yes     no
      |       |
    knowledge safe_failure
      |
   explanation
      |
   evaluation
      |
     pass?
    /     \
  yes      no
   |        |
  END    safe_failure
            |
           END
    """

    def __init__(
        self,
        prediction_client: PredictionMCPClient | None = None,
        knowledge_agent: KnowledgeAgent | None = None,
        explanation_agent: ExplanationAgent | None = None,
        evaluator: ClinRAGEvaluator | None = None,
    ):
        self.prediction_client = prediction_client or PredictionMCPClient()
        self.knowledge_agent = knowledge_agent or KnowledgeAgent()
        self.explanation_agent = explanation_agent or ExplanationAgent()
        self.evaluator = evaluator or ClinRAGEvaluator()

        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ClinRAGState)

        workflow.add_node("prediction", self._prediction_node)
        workflow.add_node("knowledge", self._knowledge_node)
        workflow.add_node("explanation", self._explanation_node)
        workflow.add_node("evaluation", self._evaluation_node)
        workflow.add_node("safe_failure", self._safe_failure_node)

        workflow.add_edge(START, "prediction")

        workflow.add_conditional_edges(
            "prediction",
            self._route_after_prediction,
            {
                "knowledge": "knowledge",
                "safe_failure": "safe_failure",
            },
        )

        workflow.add_edge("knowledge", "explanation")
        workflow.add_edge("explanation", "evaluation")

        workflow.add_conditional_edges(
            "evaluation",
            self._route_after_evaluation,
            {
                "end": END,
                "safe_failure": "safe_failure",
            },
        )

        workflow.add_edge("safe_failure", END)

        return workflow.compile()

    async def _prediction_node(self, state: ClinRAGState) -> dict:
        try:
            prediction = await self.prediction_client.predict_patient(
                state["features"]
            )

            return {
                "prediction": prediction,
                "errors": [],
                "safe_failure": False,
            }

        except Exception as exc:
            return {
                "errors": [
                    f"Prediction step failed: {type(exc).__name__}: {exc}"
                ],
                "safe_failure": True,
            }

    def _route_after_prediction(self, state: ClinRAGState) -> str:
        if state.get("prediction") and not state.get("errors"):
            return "knowledge"

        return "safe_failure"

    def _knowledge_node(self, state: ClinRAGState) -> dict:
        try:
            evidence = self.knowledge_agent.retrieve(
                state["question"],
                top_k=5,
            )

            return {
                "evidence": [
                    item.model_dump() for item in evidence
                ]
            }

        except Exception as exc:
            return {
                "errors": state.get("errors", [])
                + [f"Knowledge step failed: {type(exc).__name__}: {exc}"],
                "evidence": [],
            }

    def _explanation_node(self, state: ClinRAGState) -> dict:
        prediction_data = state.get("prediction")

        if not prediction_data:
            return {
                "errors": state.get("errors", [])
                + ["Explanation step skipped because prediction is missing."]
            }

        try:
            prediction = PredictionResult.model_validate(
                prediction_data
            )

            evidence = [
                EvidenceChunk.model_validate(item)
                for item in state.get("evidence", [])
            ]

            response = self.explanation_agent.explain(
                prediction=prediction,
                evidence=evidence,
                question=state["question"],
            )

            return {
                "answer": response.answer,
                "grounded": response.grounded,
            }

        except Exception as exc:
            return {
                "errors": state.get("errors", [])
                + [
                    f"Explanation step failed: "
                    f"{type(exc).__name__}: {exc}"
                ]
            }

    def _evaluation_node(self, state: ClinRAGState) -> dict:
        prediction_data = state.get("prediction")

        if not prediction_data:
            return {
                "evaluation_passed": False,
                "safe": False,
                "relevant": False,
                "grounded": False,
                "evaluation_issues": [
                    "Evaluation skipped because prediction is missing."
                ],
            }

        prediction = PredictionResult.model_validate(
            prediction_data
        )

        evidence = [
            EvidenceChunk.model_validate(item)
            for item in state.get("evidence", [])
        ]

        response = AgentResponse(
            answer=state.get("answer", ""),
            grounded=state.get("grounded", False),
            evidence_used=evidence,
        )

        evaluation = self.evaluator.evaluate(
            prediction=prediction,
            response=response,
            require_evidence=True,
        )

        return {
            "evaluation_passed": evaluation.passed,
            "safe": evaluation.safe,
            "relevant": evaluation.relevant,
            "grounded": evaluation.grounded,
            "evaluation_issues": evaluation.issues,
        }

    def _route_after_evaluation(self, state: ClinRAGState) -> str:
        if (
            state.get("evaluation_passed")
            and state.get("safe")
            and state.get("relevant")
            and state.get("grounded")
        ):
            return "end"

        return "safe_failure"

    def _safe_failure_node(self, state: ClinRAGState) -> dict:
        errors = list(state.get("errors", []))
        evaluation_issues = list(
            state.get("evaluation_issues", [])
        )

        reasons = errors + evaluation_issues

        if reasons:
            message = (
                "The ClinRAG workflow could not produce a validated "
                "evidence-grounded response."
            )
        else:
            message = (
                "The ClinRAG workflow stopped because the response "
                "did not pass the required safety and grounding checks."
            )

        return {
            "answer": message,
            "grounded": False,
            "safe_failure": True,
        }

    async def run(
        self,
        question: str,
        features: dict[str, float],
    ) -> ClinRAGState:

        initial_state: ClinRAGState = {
            "question": question,
            "features": features,
            "errors": [],
            "evaluation_issues": [],
            "safe_failure": False,
        }

        return await self.graph.ainvoke(initial_state)