from __future__ import annotations

import json
from pathlib import Path
from typing import BinaryIO

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"


def required_feature_columns(
    metadata_path: str | Path = METADATA_PATH,
) -> list[str]:
    with Path(metadata_path).open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    return list(metadata["feature_columns"])


def read_patient_csv(
    file: str | Path | BinaryIO,
    metadata_path: str | Path = METADATA_PATH,
) -> pd.DataFrame:
    dataframe = pd.read_csv(file)
    required = required_feature_columns(metadata_path)
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "CSV is missing required model features: "
            + ", ".join(missing)
        )

    if dataframe.empty:
        raise ValueError("CSV must contain at least one patient row.")

    numeric = dataframe[required].apply(pd.to_numeric, errors="coerce")
    invalid_columns = [
        column for column in required if numeric[column].isna().any()
    ]
    if invalid_columns:
        raise ValueError(
            "CSV contains missing or non-numeric values in model features: "
            + ", ".join(invalid_columns)
        )

    dataframe.loc[:, required] = numeric
    return dataframe


def row_to_features(
    dataframe: pd.DataFrame,
    row_index: int,
    metadata_path: str | Path = METADATA_PATH,
) -> dict[str, float]:
    required = required_feature_columns(metadata_path)
    if row_index < 0 or row_index >= len(dataframe):
        raise IndexError("Selected patient row is out of range.")

    row = dataframe.iloc[row_index]
    return {
        column: float(row[column])
        for column in required
        if pd.notna(row[column])
    }
