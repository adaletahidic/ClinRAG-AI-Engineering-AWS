from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from app.config import (
    ALLOWED_ORIGINS,
    LOG_LEVEL,
    MAX_REQUEST_BODY_BYTES,
    WORKFLOW_TIMEOUT_SECONDS,
)
from app.graph.clinrag_graph import ClinRAGGraph
from app.observability import emit_workflow_event
import asyncio
import logging


logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))


class RequestLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_body_bytes: int):
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_body_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "request_id": request.headers.get(
                        "x-request-id",
                        str(uuid4()),
                    ),
                    "error": "Request body exceeds configured limit.",
                },
            )
        return await call_next(request)


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
    app.add_middleware(
        RequestLimitMiddleware,
        max_body_bytes=MAX_REQUEST_BODY_BYTES,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=ALLOWED_ORIGINS != ["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
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
            result = await asyncio.wait_for(
                get_graph().run(
                    question=payload.question,
                    features=payload.features,
                    request_id=request_id,
                ),
                timeout=WORKFLOW_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            if isinstance(exc, asyncio.TimeoutError):
                message = "Workflow execution exceeded the configured timeout."
                emit_workflow_event(
                    {
                        "request_id": request_id,
                        "step": "api",
                        "status": "timeout",
                        "error": message,
                    }
                )
            else:
                message = (
                    f"Workflow execution failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            raise HTTPException(
                status_code=500,
                detail={
                    "request_id": request_id,
                    "error": message,
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
