from __future__ import annotations

from app.domain.predictor import BiLSTMPredictor
from app.domain.schemas import PatientFeatures, PredictionResult


class PredictionAgent:
    """
    Agent responsible for obtaining a structured prediction
    from the trained BiLSTM prediction service.

    The agent does not implement model logic itself.
    It delegates inference to BiLSTMPredictor.
    """

    def __init__(
        self,
        predictor: BiLSTMPredictor | None = None,
    ):
        self.predictor = predictor or BiLSTMPredictor()

    def predict(
        self,
        patient: PatientFeatures,
    ) -> PredictionResult:
        """
        Run prediction for a structured patient profile.
        """

        return self.predictor.predict(
            patient.features
        )

    def get_model_metadata(self):
        """
        Return metadata describing the prediction model.
        """

        return self.predictor.get_metadata()