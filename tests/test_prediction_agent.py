from app.agents.prediction_agent import PredictionAgent
from app.domain.predictor import BiLSTMPredictor
from app.domain.schemas import PatientFeatures


def test_prediction_agent():

    predictor = BiLSTMPredictor("models")
    agent = PredictionAgent(predictor)

    patient = PatientFeatures(
        features={
            feature: predictor.feature_medians.get(
                feature,
                0.0,
            )
            for feature in predictor.feature_columns
        }
    )

    result = agent.predict(patient)

    assert result.prediction in (0, 1)
    assert 0.0 <= result.probability <= 1.0
    assert result.risk_group in ("High", "Low")