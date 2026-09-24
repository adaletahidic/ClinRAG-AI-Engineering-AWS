from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app


class FakeGraph:
    async def run(
        self,
        question: str,
        features: dict[str, float],
        request_id: str,
    ) -> dict:
        return {
            "request_id": request_id,
            "answer": "Validated explanation.",
            "grounded": True,
            "safe_failure": False,
            "prediction": {
                "prediction": 1,
                "probability": 0.92,
                "risk_group": "High",
            },
            "evidence": [],
            "evaluation_passed": True,
            "safe": True,
            "relevant": True,
            "evaluation_issues": [],
            "errors": [],
            "trace": [],
            "workflow_latency_ms": 1.2,
        }


def test_health_endpoint():
    client = TestClient(create_app(FakeGraph()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_workflow_endpoint_propagates_request_id():
    client = TestClient(create_app(FakeGraph()))

    response = client.post(
        "/workflow",
        headers={"x-request-id": "api-request-1"},
        json={
            "question": "Explain the prediction.",
            "features": {"feature_1": 1.0},
        },
    )

    assert response.status_code == 200
    assert response.json()["request_id"] == "api-request-1"
    assert response.json()["evaluation_passed"] is True


def test_predict_and_explain_aliases_use_same_workflow_contract():
    client = TestClient(create_app(FakeGraph()))
    payload = {
        "question": "Explain the prediction.",
        "features": {"feature_1": 1.0},
    }

    assert client.post("/predict", json=payload).status_code == 200
    assert client.post("/explain", json=payload).status_code == 200


def test_workflow_rejects_empty_question():
    client = TestClient(create_app(FakeGraph()))

    response = client.post(
        "/workflow",
        json={"question": "", "features": {}},
    )

    assert response.status_code == 422
