from __future__ import annotations

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """
    Structured patient/sample features expected by the predictive model.

    The actual feature list will later be loaded from model metadata.
    This schema provides the initial API contract.
    """

    features: dict[str, float]


class PredictionResult(BaseModel):
    """
    Authoritative output of the predictive ML model.

    Important:
    The LLM must not create or modify these values.
    """

    prediction: int = Field(
        ge=0,
        le=1,
        description="Binary model prediction."
    )

    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Predicted probability for the positive class."
    )

    risk_group: str

    model_name: str

    model_version: str


class EvidenceChunk(BaseModel):
    """
    A single piece of retrieved knowledge-base evidence.
    """

    source: str

    page: int | None = None

    text: str

    relevance: float = Field(
        ge=0.0,
        le=1.0
    )


class ModelMetadata(BaseModel):
    """
    Metadata describing the model artifact and evaluation results.
    """

    model_name: str

    model_version: str

    dataset: str

    feature_columns: list[str]

    threshold: float = Field(
        ge=0.0,
        le=1.0
    )

    auc: float | None = None

    accuracy: float | None = None

    precision: float | None = None

    recall: float | None = None

    f1: float | None = None


class ExplanationRequest(BaseModel):
    """
    Input contract for the explanation agent.
    """

    question: str

    prediction: PredictionResult | None = None

    evidence: list[EvidenceChunk] = Field(
        default_factory=list
    )


class AgentResponse(BaseModel):
    """
    Standard response returned by an AI agent.
    """

    answer: str

    grounded: bool

    evidence_used: list[EvidenceChunk] = Field(
        default_factory=list
    )


class EvaluationResult(BaseModel):
    """
    Quality and safety evaluation of an agent response.
    """

    passed: bool

    grounded: bool

    safe: bool

    relevant: bool

    issues: list[str] = Field(
        default_factory=list
    )