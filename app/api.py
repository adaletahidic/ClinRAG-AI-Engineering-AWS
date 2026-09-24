from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app.graph.clinrag_graph import ClinRAGGraph


class WorkflowRequest(BaseModel):
    question: str = Field(min_length=1)
    features: dict[str, float] = Field(default_factory=dict)


class WorkflowResponse(BaseModel):
    request_id: str
    answer: str | None = None
    grounded: bool = False
    safe_failure: bool = False
    prediction: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    evaluation_passed: bool = False
    safe: bool = False
    relevant: bool = False
    evaluation_issues: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    workflow_latency_ms: float | None = None


class ErrorResponse(BaseModel):
    request_id: str
    error: str


def create_app(graph: ClinRAGGraph | None = None) -> FastAPI:
    """Create the HTTP boundary while keeping the graph injectable in tests."""
    app = FastAPI(
        title="ClinRAG API",
        version="1.0.0",
    )

    workflow_graph = graph

    def get_graph() -> ClinRAGGraph:
        nonlocal workflow_graph
        if workflow_graph is None:
            workflow_graph = ClinRAGGraph()
        return workflow_graph

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    async def execute_workflow(
        payload: WorkflowRequest,
        request_id: str,
    ) -> WorkflowResponse:
        try:
            result = await get_graph().run(
                question=payload.question,
                features=payload.features,
                request_id=request_id,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "request_id": request_id,
                    "error": (
                        f"Workflow execution failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                },
            ) from exc

        result["request_id"] = request_id
        return WorkflowResponse.model_validate(result)

    @app.post(
        "/workflow",
        response_model=WorkflowResponse,
        responses={500: {"model": ErrorResponse}},
    )
    async def workflow(
        payload: WorkflowRequest,
        x_request_id: str | None = Header(default=None),
    ) -> WorkflowResponse:
        return await execute_workflow(
            payload,
            x_request_id or str(uuid4()),
        )

    @app.post(
        "/predict",
        response_model=WorkflowResponse,
        responses={500: {"model": ErrorResponse}},
    )
    async def predict(
        payload: WorkflowRequest,
        x_request_id: str | None = Header(default=None),
    ) -> WorkflowResponse:
        return await execute_workflow(
            payload,
            x_request_id or str(uuid4()),
        )

    @app.post(
        "/explain",
        response_model=WorkflowResponse,
        responses={500: {"model": ErrorResponse}},
    )
    async def explain(
        payload: WorkflowRequest,
        x_request_id: str | None = Header(default=None),
    ) -> WorkflowResponse:
        return await execute_workflow(
            payload,
            x_request_id or str(uuid4()),
        )

    return app


app = create_app()
