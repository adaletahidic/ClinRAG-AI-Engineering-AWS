from app.domain.predictor import BiLSTMPredictor


def test_predictor_loads():
    predictor = BiLSTMPredictor("models")

    assert predictor.model is not None
    assert predictor.scaler is not None
    assert predictor.feature_columns
    assert 0.0 <= predictor.threshold <= 1.0


def test_prediction():
    predictor = BiLSTMPredictor("models")

    features = {
        feature: predictor.feature_medians.get(
            feature,
            0.0,
        )
        for feature in predictor.feature_columns
    }

    result = predictor.predict(features)

    assert result.prediction in (0, 1)
    assert 0.0 <= result.probability <= 1.0
    assert result.risk_group in ("High", "Low")