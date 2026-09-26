"""
Phase 3 - LLM Integration & AI Insights  (Gemini free-tier version)
---------------------------------------------------------------------
Two functions the Streamlit app (Phase 4) calls:
  1. generate_insight(context)              -> plain-English paragraph, no user question
  2. answer_question(context, question)     -> answers a specific question about the well

WHY THIS VERSION USES GEMINI INSTEAD OF CLAUDE:
Anthropic's API is pay-as-you-go with no reliable ongoing free tier. Google's
Gemini API has a genuine free tier (no credit card required) for models like
gemini-2.5-flash, which is enough for this project's needs. Everything else
about the design - the system prompt, the guardrails, the two function
signatures the app calls - is UNCHANGED from the original design, so Phase 4
(app.py) does not need to change at all.

SETUP (run this in your own environment, not here):
    pip install google-genai
    Get a free API key at https://aistudio.google.com/apikey (no card needed)
    export GOOGLE_API_KEY="your-key-here"

Design principles (unchanged from the original design):
- The context dict (from context_builder.py) is the ONLY source of numbers the
  model is allowed to use. The system prompt explicitly forbids inventing
  numbers not present in the context - this is the single most important
  guardrail for a trustworthy insights layer.
- Model output is kept short (a few sentences) - this is a decision-support
  tool for an engineer, not a report generator.
- Every response is grounded in the SAME context object shown to the model,
  so the Streamlit app can display "based on data as of <date>" next to the
  answer for traceability.
"""
import os
import json
from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.1-flash-lite"  # confirmed free tier as of Sept 2026 (ai.google.dev/pricing); gemini-2.5-flash was retired

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

SYSTEM_PROMPT = """You are a petroleum production engineering assistant embedded in a \
well-forecasting app. You will be given a JSON object with pre-computed numbers about \
one well's production forecast. Follow these rules strictly:

1. Use ONLY the numbers in the provided context. Never invent, estimate, or round in a \
way that changes a figure's meaning. If something isn't in the context, say you don't \
have that data rather than guessing.
2. Explain the ARPS forecast as the primary/trusted forecast. Mention the ML model \
(XGBoost) only as supporting context about which factors are influencing the well's \
recent behavior, never as a competing forecast number - the context tells you Arps \
outperforms it on this well.
3. Write for a petroleum engineer, not a general audience - it's fine to use terms \
like decline rate, drawdown, bopd without defining them.
4. Keep answers concise: 3-5 sentences for a general insight, 1-3 for a direct question.
5. If the deviation between actual and forecast is small (a few percent), say the well \
is performing in line with forecast - don't manufacture concern out of noise.
"""


def _call_gemini(user_message):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=400,
            temperature=0.3,
        ),
    )
    return response.text


def generate_insight(context: dict) -> str:
    """Unprompted summary - what the app shows by default when a well is selected."""
    user_message = (
        "Here is the current forecast context for a well:\n\n"
        f"{json.dumps(context, indent=2)}\n\n"
        "Write a short plain-English insight paragraph a reservoir engineer would find "
        "useful at a glance: how the well is performing versus forecast, and whether "
        "anything about the recent trend is worth a closer look."
    )
    return _call_gemini(user_message)


def answer_question(context: dict, question: str) -> str:
    """Grounded Q&A - e.g. 'why is Well X declining faster than expected?'"""
    user_message = (
        "Here is the current forecast context for a well:\n\n"
        f"{json.dumps(context, indent=2)}\n\n"
        f"Engineer's question: {question}\n\n"
        "Answer using only the numbers above. If the question asks about something "
        "not covered by this context (e.g. a different well, or a data field not "
        "present), say so explicitly rather than guessing."
    )
    return _call_gemini(user_message)


if __name__ == "__main__":
    # Quick manual test once you have GOOGLE_API_KEY set.
    from context_builder import build_insight_context

    ctx = build_insight_context(
        cleaned_csv_path="volve_f14_cleaned.csv",
        arps_test_preds_path="arps_test_preds.csv",
        arps_params={"qi": 568.99, "di": 0.00182},
        arps_test_r2=0.771,
        arps_test_mape=0.105,
        ml_test_r2=-0.061,
        top_ml_features=["yesterday's rate (dominant)", "7-day rolling average rate", "downhole temperature"],
    )

    print("=== Auto insight ===")
    print(generate_insight(ctx))

    print("\n=== Q&A ===")
    print(answer_question(ctx, "Why is this well declining faster than expected?"))
