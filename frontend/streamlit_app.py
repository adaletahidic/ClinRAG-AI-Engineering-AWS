from __future__ import annotations

import os

import requests
import streamlit as st

from app.csv_validation import read_patient_csv, row_to_features


API_URL = os.getenv("CLINRAG_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="ClinRAG Clinical AI Demo",
    page_icon="🩺",
    layout="wide",
)

st.title("ClinRAG Clinical AI Decision Support")
st.caption(
    "The model prediction is authoritative. The LLM only explains it "
    "using retrieved evidence and is not a medical diagnosis."
)

with st.sidebar:
    st.subheader("Configuration")
    st.code(API_URL)
    st.info(
        "Use synthetic or de-identified data. Do not upload real patient "
        "records to this local demo."
    )

uploaded_file = st.file_uploader(
    "Upload patient CSV",
    type=["csv"],
    help="CSV must contain the 30 feature columns from models/metadata.json.",
)

if uploaded_file is None:
    st.info("Upload one of the patient example CSV files to begin.")
    st.stop()

try:
    dataframe = read_patient_csv(uploaded_file)
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Patient rows")
st.dataframe(dataframe, use_container_width=True)

row_index = st.selectbox(
    "Select patient row",
    options=list(range(len(dataframe))),
    format_func=lambda index: f"Row {index + 1}",
)

question = st.text_area(
    "Question for the explanation agent",
    value="Explain the model prediction using the available clinical evidence.",
)

if st.button("Run prediction and explanation", type="primary"):
    try:
        features = row_to_features(dataframe, row_index)
        response = requests.post(
            f"{API_URL}/workflow",
            json={
                "question": question,
                "features": features,
            },
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        st.stop()
    except (ValueError, IndexError) as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Prediction")
    prediction = result.get("prediction")
    if prediction:
        columns = st.columns(4)
        columns[0].metric("Prediction", prediction.get("prediction", "N/A"))
        columns[1].metric(
            "Probability",
            f"{prediction.get('probability', 0):.4f}",
        )
        columns[2].metric("Risk group", prediction.get("risk_group", "N/A"))
        columns[3].metric("Model", prediction.get("model_name", "N/A"))
    else:
        st.warning("No prediction was returned.")

    if result.get("safe_failure"):
        st.error(result.get("answer", "Workflow stopped safely."))
    else:
        st.subheader("Evidence-grounded explanation")
        st.write(result.get("answer", "No explanation returned."))

    status_columns = st.columns(4)
    status_columns[0].metric(
        "Evaluation",
        "PASS" if result.get("evaluation_passed") else "FAIL",
    )
    status_columns[1].metric("Grounded", str(result.get("grounded", False)))
    status_columns[2].metric("Safe", str(result.get("safe", False)))
    status_columns[3].metric(
        "Latency",
        f"{result.get('workflow_latency_ms', 0):.2f} ms",
    )

    evidence = result.get("evidence", [])
    if evidence:
        st.subheader("Retrieved evidence")
        for index, item in enumerate(evidence, start=1):
            with st.expander(
                f"Evidence {index}: {item.get('source', 'unknown')}"
            ):
                st.write(f"Relevance: {item.get('relevance', 0):.4f}")
                st.write(item.get("text", ""))

    issues = result.get("evaluation_issues", []) + result.get("errors", [])
    if issues:
        st.warning("Workflow details: " + " | ".join(issues))

    with st.expander("Workflow trace"):
        st.json(result.get("trace", []))
