from __future__ import annotations

from app.domain.schemas import (
    AgentResponse,
    EvidenceChunk,
    PredictionResult,
)
from app.llm.base import LLMProvider
from app.llm.factory import get_llm_provider


class ExplanationAgent:
    """
    Generates a grounded explanation for an existing ML prediction.

    The ExplanationAgent does not create or modify the prediction.
    It only explains the prediction using the supplied evidence.
    """

    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm

    def explain(
        self,
        prediction: PredictionResult,
        evidence: list[EvidenceChunk],
        question: str,
    ) -> AgentResponse:

        # Dependency injection is used in tests.
        # In production, the configured provider is created automatically.
        if self.llm is None:
            self.llm = get_llm_provider()

        return self._llm_explanation(
            prediction=prediction,
            evidence=evidence,
            question=question,
        )

    def _llm_explanation(
        self,
        prediction: PredictionResult,
        evidence: list[EvidenceChunk],
        question: str,
    ) -> AgentResponse:

        evidence_text = "\n\n".join(
            [
                (
                    f"[Evidence {index}]\n"
                    f"Source: {item.source}\n"
                    f"Page: {item.page}\n"
                    f"Relevance: {item.relevance:.4f}\n"
                    f"Text: {item.text}"
                )
                for index, item in enumerate(evidence, start=1)
            ]
        )

        if not evidence_text:
            evidence_text = "No external evidence was retrieved."

        system_prompt = """
You are an explanation component in a clinical AI decision-support system.

Your role is strictly limited to explaining an existing machine-learning
prediction using the retrieved evidence.

Rules:
1. Never change or reinterpret the prediction.
2. Never change the probability.
3. Never change the risk group.
4. Never invent evidence.
5. Use only the supplied evidence for evidence-based claims.
6. Do not present the prediction as a medical diagnosis.
7. Do not recommend treatment.
8. If there is no evidence, explicitly state that no evidence-grounded
   clinical explanation can be provided.
9. Keep the explanation concise and traceable.
10. The explanation layer must never create or modify the prediction.
"""

        user_prompt = f"""
Question:
{question}

Model prediction:
{prediction.prediction}

Predicted probability:
{prediction.probability:.4f}

Risk group:
{prediction.risk_group}

Model:
{prediction.model_name}

Model version:
{prediction.model_version}

Retrieved evidence:
{evidence_text}
"""

        answer = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # The grounding status is determined by the actual evidence
        # supplied to the agent, not by the LLM's own claim.
        return AgentResponse(
            answer=answer,
            grounded=bool(evidence),
            evidence_used=evidence,
        )