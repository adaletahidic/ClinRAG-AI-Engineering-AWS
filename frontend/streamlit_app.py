from __future__ import annotations

import os

import requests
import streamlit as st

from app.csv_validation import read_patient_csv, row_to_features


API_URL = os.getenv("CLINRAG_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="ClinRAG | Clinical AI",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background:#f2f8fa; color:#173f52; }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div { color:#173f52; }
    [data-testid="stHeader"] { background:#e7f3f5 !important; height:2.5rem; }
    [data-testid="stToolbar"] { background:transparent !important; }
    [data-testid="stDecoration"] { background:#2b9c9a !important; height:3px; }
    [data-testid="stAppViewContainer"] { background:#f2f8fa; }
    [data-testid="stMainBlockContainer"] { padding-top:1rem; }
    .block-container { max-width:1240px; padding-top:4.5rem; }
    .brand { display:flex; align-items:center; gap:.75rem; }
    .brand-mark { background:#dff1f0; color:#087f83; border-radius:14px; padding:.55rem .7rem; font-size:1.8rem; }
    .brand-title { color:#123b50 !important; font-size:2.05rem; font-weight:750; letter-spacing:-.02em; }
    .subtitle { color:#607482 !important; margin-top:-.7rem; }
    h1,h2,h3,h4,p,label,[data-testid="stCaptionContainer"] { color:#173f52 !important; }
    [data-testid="stMetric"] { background:#ffffff; border:1px solid #cfe2e5; padding:1rem; border-radius:12px; box-shadow:0 2px 8px rgba(22,59,77,.06); }
    [data-testid="stMetricLabel"] p { color:#52707b !important; font-weight:600; }
    [data-testid="stMetricValue"] { color:#163b4d !important; }
    [data-testid="stSidebar"] { background:#e6f2f4; border-right:1px solid #c8e0e3; }
    [data-testid="stFileUploader"] section { background:#ffffff; border:1px dashed #91b9bd; }
    [data-testid="stFileUploader"] section div,
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section small { color:#163b4d !important; }
    .stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div { background:#ffffff; color:#163b4d; }
    .stTextArea textarea, .stTextInput input { caret-color:#087f83; }
    .stButton > button {
        background:#087f83 !important;
        color:#ffffff !important;
        border:1px solid #087f83 !important;
        border-radius:9px !important;
        font-weight:700 !important;
    }
    .stButton > button:hover { background:#066568 !important; color:#ffffff !important; }
    .stButton > button p, .stButton > button span { color:#ffffff !important; }
    [data-baseweb="select"] * { color:#173f52 !important; }
    [data-baseweb="popover"] { background:#ffffff !important; }
    .stChatMessage { background:#ffffff; border:1px solid #cfe2e5; }
    [data-testid="stAlert"] p { color:#173f52 !important; }
    </style>
    <div class="brand"><div class="brand-mark">🩺</div>
    <div class="brand-title">ClinRAG Clinical AI</div></div>
    <p class="subtitle">Evidence-grounded decision support for breast-cancer risk modelling</p>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "Decision-support prototype only. The model output is not a diagnosis "
    "and must be reviewed by a qualified clinician."
)

with st.sidebar:
    st.header("Knowledge base")
    st.caption("Add PDF, TXT or Markdown sources for this running session.")
    knowledge_files = st.file_uploader(
        "Upload grounding documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="knowledge_files",
    )
    if st.button("Add documents to RAG"):
        if not knowledge_files:
            st.info("Select at least one document first.")
        else:
            for knowledge_file in knowledge_files:
                try:
                    upload_response = requests.post(
                        f"{API_URL}/knowledge/upload",
                        files={
                            "file": (
                                knowledge_file.name,
                                knowledge_file.getvalue(),
                                knowledge_file.type,
                            )
                        },
                        timeout=60,
                    )
                    upload_response.raise_for_status()
                    details = upload_response.json()
                    st.success(
                        f"{details['filename']}: "
                        f"{details['chunks_added']} chunks added"
                    )
                except requests.RequestException as exc:
                    st.error(f"Could not add {knowledge_file.name}: {exc}")
    st.divider()
    st.subheader("Connection")
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
    st.info("Upload a patient CSV to begin the prediction and chat workflow.")
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

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

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
        st.session_state.workflow_result = result
        st.session_state.chat_features = features
        st.session_state.chat_messages = [
            {"role": "assistant", "content": result.get("answer", "")}
        ]
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        st.stop()
    except (ValueError, IndexError) as exc:
        st.error(str(exc))
        st.stop()

if "workflow_result" in st.session_state:
    result = st.session_state.workflow_result
    st.subheader("Prediction and validated explanation")
    if result.get("safe_failure"):
        st.error(
            "The response was stopped because it did not pass the "
            "evidence-grounding and safety checks."
        )
        failure_details = result.get("evaluation_issues", []) + result.get(
            "errors", []
        )
        if failure_details:
            st.warning("Reason: " + " | ".join(failure_details))
    elif result.get("evaluation_passed"):
        st.success(
            "Validated response: prediction, retrieved evidence, "
            "explanation and safety checks passed."
        )
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

    st.divider()
    st.subheader("Continue the evidence-grounded chat")
    st.caption(
        "Follow-up questions use the same selected patient features and the "
        "current knowledge base. The original model prediction is recalculated "
        "by the backend and cannot be changed by the chat."
    )
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    follow_up = st.chat_input(
        "Ask a follow-up about the prediction or retrieved evidence"
    )
    if follow_up:
        st.session_state.chat_messages.append(
            {"role": "user", "content": follow_up}
        )
        try:
            chat_response = requests.post(
                f"{API_URL}/workflow",
                json={
                    "question": follow_up,
                    "features": st.session_state.chat_features,
                },
                timeout=180,
            )
            chat_response.raise_for_status()
            chat_result = chat_response.json()
            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": chat_result.get(
                        "answer",
                        "No validated response was returned.",
                    ),
                }
            )
            st.session_state.workflow_result = chat_result
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Chat request failed: {exc}")
