from app.mcp.prediction_server import (
    predict_patient,
    get_prediction_model_metadata,
)
from app.domain.predictor import BiLSTMPredictor


def test_mcp_prediction_tool():

    predictor = BiLSTMPredictor("models")

    features = {
        feature: predictor.feature_medians.get(
            feature,
            0.0,
        )
        for feature in predictor.feature_columns
    }

    result = predict_patient(features)

    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_group"] in ("High", "Low")
    assert result["model_name"] == "Bayesian-BiLSTM"


def test_mcp_metadata_tool():

    result = get_prediction_model_metadata()

    assert result["model_name"] == "Bayesian-BiLSTM"
    assert result["dataset"] == (
        "Wisconsin Diagnostic Breast Cancer"
    )
    assert result["feature_columns"]