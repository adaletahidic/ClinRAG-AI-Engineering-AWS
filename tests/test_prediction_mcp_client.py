from __future__ import annotations

import pytest

from app.mcp.prediction_client import (
    PredictionMCPClient,
)


@pytest.mark.asyncio
async def test_prediction_mcp_client():

    client = PredictionMCPClient()

    tools = await client.list_tools()

    assert "predict_patient" in tools
    assert "get_prediction_model_metadata" in tools


@pytest.mark.asyncio
async def test_prediction_mcp_client_prediction():

    client = PredictionMCPClient()

    metadata = await client.get_model_metadata()

    features = {
        feature: 0.0
        for feature in metadata["feature_columns"]
    }

    result = await client.predict_patient(
        features
    )

    assert result["prediction"] in (0, 1)

    assert 0.0 <= result["probability"] <= 1.0

    assert result["risk_group"] in (
        "High",
        "Low",
    )

    assert result["model_name"] == "Bayesian-BiLSTM"

    assert result["model_version"] == "1.0"