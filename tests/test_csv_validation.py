from __future__ import annotations

from io import StringIO

import pytest

from app.csv_validation import read_patient_csv, row_to_features


def test_csv_validation_reads_model_features():
    csv = StringIO(
        "radius_mean,texture_mean\n"
        "13.37,18.84\n"
    )

    with pytest.raises(ValueError, match="missing required model features"):
        read_patient_csv(csv)


def test_row_to_features_rejects_invalid_row():
    csv = StringIO(
        "radius_mean,texture_mean\n"
        "13.37,18.84\n"
    )

    with pytest.raises(ValueError):
        dataframe = read_patient_csv(csv)
        row_to_features(dataframe, 0)
