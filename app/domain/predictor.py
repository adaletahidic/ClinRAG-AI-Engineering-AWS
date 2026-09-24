from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from app.domain.schemas import ModelMetadata, PredictionResult


class BiLSTMPredictor:
    """
    Adapter around the previously trained Bayesian-optimized BiLSTM model.

    The predictor is responsible only for:
    - loading model artifacts
    - validating and preparing input features
    - running inference
    - returning a structured prediction result
    """

    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)

        self.model_path = self.model_dir / "bilstm_model.keras"
        self.scaler_path = self.model_dir / "scaler.joblib"
        self.metadata_path = self.model_dir / "metadata.json"

        self._validate_artifacts()

        self.model = load_model(
            self.model_path,
            compile=False,
        )

        self.scaler = joblib.load(self.scaler_path)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.feature_columns = self.metadata["feature_columns"]
        self.threshold = self.metadata.get("threshold", 0.5)
        self.feature_medians = self.metadata.get(
            "feature_medians",
            {},
        )

    def _validate_artifacts(self) -> None:
        required_files = [
            self.model_path,
            self.scaler_path,
            self.metadata_path,
        ]

        missing = [
            str(path)
            for path in required_files
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing model artifacts: {missing}"
            )

    def _prepare_features(
        self,
        features: dict[str, float],
    ) -> np.ndarray:

        df = pd.DataFrame([features])

        missing_features = [
            feature
            for feature in self.feature_columns
            if feature not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}"
            )

        df = df[self.feature_columns]

        df = df.apply(
            pd.to_numeric,
            errors="coerce",
        )

        for feature in self.feature_columns:
            if df[feature].isna().any():
                median = self.feature_medians.get(feature)

                if median is None:
                    raise ValueError(
                        f"Missing value for '{feature}' "
                        "and no median is available."
                    )

                df[feature] = df[feature].fillna(median)

        scaled = self.scaler.transform(df)

        # Original model expects:
        # (samples, timesteps, features)
        #
        # For this tabular formulation:
        # timesteps = 1
        X = scaled.reshape(
            (scaled.shape[0], 1, scaled.shape[1])
        )

        return X

    def predict(
        self,
        features: dict[str, float],
    ) -> PredictionResult:

        X = self._prepare_features(features)

        probability = float(
            self.model.predict(
                X,
                verbose=0,
            )[0][0]
        )

        prediction = int(
            probability >= self.threshold
        )

        risk_group = (
            "High"
            if prediction == 1
            else "Low"
        )

        return PredictionResult(
            prediction=prediction,
            probability=probability,
            risk_group=risk_group,
            model_name="Bayesian-BiLSTM",
            model_version="1.0",
        )

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name="Bayesian-BiLSTM",
            model_version="1.0",
            dataset="Wisconsin Diagnostic Breast Cancer",
            feature_columns=self.feature_columns,
            threshold=self.threshold,
        )