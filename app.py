"""
Phase 4 - Complete AI Application (Streamlit)
------------------------------------------------
Well 15/9-F-14 Production Forecast & AI Insights App.

RUN LOCALLY:
    pip install streamlit pandas numpy plotly google-genai
    export GOOGLE_API_KEY="your-key-here"   (free key: https://aistudio.google.com/apikey)
    streamlit run app.py

FILES THIS APP EXPECTS IN THE SAME FOLDER:
    volve_f14_cleaned.csv        (Phase 1 output)
    phase2_final_test_predictions.csv   (Phase 2 output - actual, RF, XGB, naive)
    arps_test_preds.csv          (reconstructed Phase 2 Arps output - date, oil_rate_bopd, arps_pred)
    context_builder.py           (Phase 3)
    llm_insights.py              (Phase 3)

DESIGN NOTES (why the app looks the way it does):
- Two charts, not one: a full-history overview (log scale) shows the well's
  entire 2008-2016 life so the decline is visible at true scale; a zoomed
  test-period chart (linear scale) shows Actual vs Arps vs XGBoost for the
  534-day test window, since a single linear chart across the full history
  would flatten the recent decline into an unreadable sliver near zero.
- Arps is drawn as the solid, primary forecast line. XGBoost is drawn dashed
  and labeled "context only" - this matches the framing already built into
  llm_insights.py's system prompt, so the UI and the AI's language agree
  with each other rather than contradicting.
- The insights panel and chat box call the Phase 3 functions directly and
  do not reimplement any of their logic.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from context_builder import build_insight_context
from llm_insights import generate_insight, answer_question

st.set_page_config(page_title="Well F-14 Production Forecast", layout="wide")

# ---------------------------------------------------------------------------
# Data loading (cached so re-runs on user interaction don't reread from disk)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    full_history = pd.read_csv("volve_f14_cleaned.csv", parse_dates=["date"])
    test = pd.read_csv("phase2_final_test_predictions.csv", parse_dates=["date"])
    arps = pd.read_csv("arps_test_preds.csv", parse_dates=["date"])
    test = test.merge(arps[["date", "arps_pred"]], on="date", how="left")
    return full_history, test

full_history, test = load_data()

# Fixed model-quality numbers from Phase 2 reconciliation (see Continuity
# Summary v2, Section 5). These are constants, not recomputed live, because
# they describe the trained models' test performance, not something the
# app derives at runtime.
ARPS_PARAMS = {"qi": 568.99, "di": 0.00182}
ARPS_TEST_R2 = 0.771
ARPS_TEST_MAPE = 0.105  # verified: full 534-day test window, excluding 4 zero-actual (shut-in) days
ML_TEST_R2 = -0.061
TOP_ML_FEATURES = [
    "yesterday's rate (dominant)",
    "7-day rolling average rate",
    "downhole temperature",
]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Well 15/9-F-14")
st.sidebar.caption("Volve field (Equinor open dataset) — single-well forecast demo")
st.sidebar.markdown(
    "**Primary forecast:** Arps decline curve\n\n"
    "**Secondary (context only):** XGBoost, pooled multi-well model"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"Arps test R² = **{ARPS_TEST_R2}**  \n"
    f"XGBoost test R² = **{ML_TEST_R2}**  \n"
    "_(full 534-day test window; see report for full model comparison)_"
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Well 15/9-F-14 — Production Forecast & AI Insights")
st.caption(
    "Arps decline curve as the trusted forecast, with a pooled ML model shown "
    "for context. An AI assistant explains the forecast and answers questions "
    "grounded strictly in the numbers below."
)

# ---------------------------------------------------------------------------
# Chart 1: Full production history (log scale)
# ---------------------------------------------------------------------------

st.subheader("Full production history (2008–2016)")

prod_history = full_history[full_history["oil_rate_bopd"] > 0]

fig_history = go.Figure()
fig_history.add_trace(go.Scatter(
    x=prod_history["date"], y=prod_history["oil_rate_bopd"],
    mode="lines", name="Oil rate (bopd)", line=dict(color="#2E5266", width=1.5),
))
fig_history.update_layout(
    yaxis_type="log", yaxis_title="Oil rate (bopd, log scale)",
    xaxis_title="Date", height=350, margin=dict(t=10, b=10),
    showlegend=False,
)
st.plotly_chart(fig_history, use_container_width=True)
st.caption(
    "Log scale is used here because the well's rate falls from over 3,000 bopd "
    "in 2008 to around 100 bopd by 2016 — a linear scale would hide the shape "
    "of the decline in later years."
)

# ---------------------------------------------------------------------------
# Chart 2: Test-period forecast comparison (linear scale)
# ---------------------------------------------------------------------------

st.subheader("Forecast comparison — test period")

fig_test = go.Figure()
fig_test.add_trace(go.Scatter(
    x=test["date"], y=test["oil_rate_bopd"],
    mode="lines", name="Actual", line=dict(color="#111111", width=2),
))
fig_test.add_trace(go.Scatter(
    x=test["date"], y=test["arps_pred"],
    mode="lines", name="Arps (primary forecast)",
    line=dict(color="#C1440E", width=2),
))
fig_test.add_trace(go.Scatter(
    x=test["date"], y=test["pred_xgb_pooled"],
    mode="lines", name="XGBoost pooled (context only)",
    line=dict(color="#8A8D91", width=1.5, dash="dash"),
))
fig_test.update_layout(
    yaxis_title="Oil rate (bopd)", xaxis_title="Date",
    height=400, margin=dict(t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_test, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Arps MAE", "16.5 bopd")
col2.metric("Arps R²", f"{ARPS_TEST_R2}")
col3.metric("XGBoost R² (context)", f"{ML_TEST_R2}")

# ---------------------------------------------------------------------------
# AI Insights panel
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("AI insights")

@st.cache_data
def get_context():
    return build_insight_context(
        cleaned_csv_path="volve_f14_cleaned.csv",
        arps_test_preds_path="arps_test_preds.csv",
        arps_params=ARPS_PARAMS,
        arps_test_r2=ARPS_TEST_R2,
        arps_test_mape=ARPS_TEST_MAPE,
        ml_test_r2=ML_TEST_R2,
        top_ml_features=TOP_ML_FEATURES,
    )

context = get_context()

if st.button("Generate insight"):
    with st.spinner("Analyzing forecast..."):
        insight_text = generate_insight(context)
    st.info(insight_text)
    st.caption(f"Based on data as of {context['as_of_date']}")

# ---------------------------------------------------------------------------
# Q&A chat box
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Ask about this well")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

question = st.chat_input("e.g. Why is this well declining faster than expected?")

if question:
    st.session_state.chat_history.append(("user", question))
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = answer_question(context, question)
        st.write(answer)
    st.session_state.chat_history.append(("assistant", answer))

st.caption(
    "Answers are grounded strictly in the forecast context above — the "
    "assistant will say so explicitly if a question falls outside that data."
)
