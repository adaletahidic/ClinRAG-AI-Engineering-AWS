from __future__ import annotations

from pathlib import Path

from mcp.server import MCPServer

from app.domain.predictor import BiLSTMPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"


mcp = MCPServer(
    "ClinRAG Prediction Server",
    version="1.0.0",
)


_predictor: BiLSTMPredictor | None = None


def get_predictor() -> BiLSTMPredictor:
    global _predictor

    if _predictor is None:
        _predictor = BiLSTMPredictor(
            model_dir=MODEL_DIR
        )

    return _predictor


@mcp.tool()
def predict_patient(
    features: dict[str, float],
) -> dict:
    """
    Run the validated BiLSTM prediction for a patient profile.

    This tool performs model inference only.
    It does not generate a clinical explanation
    and does not provide autonomous diagnosis.
    """

    predictor = get_predictor()

    result = predictor.predict(features)

    return result.model_dump()


@mcp.tool()
def get_prediction_model_metadata() -> dict:
    """
    Return metadata describing the prediction model.
    """

    predictor = get_predictor()

    metadata = predictor.get_metadata()

    return metadata.model_dump()


if __name__ == "__main__":
    mcp.run(transport="stdio")