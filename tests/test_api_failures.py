from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app


class FailingGraph:
    def __init__(self, error: Exception):
        self.error = error

    async def run(self, question, features, request_id):
        raise self.error


class SafeFailureGraph:
    def __init__(self, error_message: str):
        self.error_message = error_message

    async def run(self, question, features, request_id):
        return {
            "request_id": request_id,
            "answer": "The workflow could not complete safely.",
            "grounded": False,
            "safe_failure": True,
            "evaluation_passed": False,
            "safe": False,
            "relevant": False,
            "evaluation_issues": [self.error_message],
            "errors": [],
            "trace": [],
        }


def test_api_returns_500_for_prediction_failure():
    response = TestClient(
        create_app(FailingGraph(RuntimeError("prediction failed")))
    ).post("/workflow", json={"question": "Explain.", "features": {}})

    assert response.status_code == 500
    assert "prediction failed" in response.json()["detail"]["error"]


def test_api_preserves_retrieval_safe_failure():
    response = TestClient(
        create_app(SafeFailureGraph("Knowledge step failed"))
    ).post("/workflow", json={"question": "Explain.", "features": {}})

    assert response.status_code == 200
    assert response.json()["safe_failure"] is True
    assert response.json()["evaluation_passed"] is False


def test_api_preserves_explanation_safe_failure():
    response = TestClient(
        create_app(SafeFailureGraph("Explanation step failed"))
    ).post("/workflow", json={"question": "Explain.", "features": {}})

    assert response.status_code == 200
    assert response.json()["safe_failure"] is True


def test_api_preserves_unsafe_response_failure():
    response = TestClient(
        create_app(SafeFailureGraph("Potentially unsafe diagnostic statement detected."))
    ).post("/workflow", json={"question": "Explain.", "features": {}})

    assert response.status_code == 200
    assert response.json()["safe_failure"] is True
    assert response.json()["evaluation_issues"]


def test_api_returns_500_for_mcp_unavailability():
    response = TestClient(
        create_app(FailingGraph(ConnectionError("MCP server unavailable")))
    ).post("/workflow", json={"question": "Explain.", "features": {}})

    assert response.status_code == 500
    assert "MCP server unavailable" in response.json()["detail"]["error"]
