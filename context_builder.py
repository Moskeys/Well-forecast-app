"""
Phase 3 - Context Builder for LLM Insights
--------------------------------------------
Purpose: turn the Phase 1 (cleaned data) and Phase 2 (Arps + ML) outputs into
a SMALL, structured JSON object to feed the LLM - never the raw daily CSV.

Why this matters:
- Sending years of daily rows to an LLM is expensive and slow, and gives the
  model room to "notice" spurious patterns in noise instead of answering from
  the numbers that actually matter.
- Every number in the context below is COMPUTED here in Python, not by the
  LLM. The LLM's job in Phase 3 is to explain and answer questions using these
  numbers - never to calculate or invent them itself. This is what keeps the
  insight text trustworthy.

Usage:
    from context_builder import build_insight_context
    context = build_insight_context(
        cleaned_csv_path="volve_f14_cleaned.csv",
        arps_test_preds_path="arps_baseline_test_preds.csv",
        arps_params={"qi": 568.99, "di": 0.00182},
        arps_test_r2=0.864, arps_test_mape=0.098,
        ml_test_r2=-0.061,
        top_ml_features=["yesterday's rate", "7-day rolling average rate", "downhole temperature"],
    )
"""
import pandas as pd
import json


def build_insight_context(cleaned_csv_path, arps_test_preds_path, arps_params,
                            arps_test_r2, arps_test_mape, ml_test_r2, top_ml_features,
                            trend_window_days=30):
    final = pd.read_csv(cleaned_csv_path, parse_dates=['date'])
    arps = pd.read_csv(arps_test_preds_path, parse_dates=['date'])[['date', 'arps_pred']]

    merged = final.merge(arps, on='date', how='inner')
    merged['pct_dev_from_arps'] = (
        (merged['oil_rate_bopd'] - merged['arps_pred']) / merged['arps_pred'] * 100
    )
    latest = merged.iloc[-1]
    recent = merged.tail(trend_window_days)

    context = {
        "well_id": "15/9-F-14",
        "as_of_date": str(latest['date'].date()),
        "current_oil_rate_bopd": round(float(latest['oil_rate_bopd']), 1),
        "arps_forecast_bopd": round(float(latest['arps_pred']), 1),
        "pct_deviation_from_arps_forecast": round(float(latest['pct_dev_from_arps']), 1),
        f"avg_deviation_last_{trend_window_days}_days_pct": round(float(recent['pct_dev_from_arps'].mean()), 1),
        "arps_decline_params": {
            "qi_bopd": arps_params["qi"],
            "di_per_day": round(arps_params["di"], 6),
            "model_type": "exponential, re-anchored to most recent 365 days of production",
        },
        "arps_model_fit_quality": {
            "test_r2": round(arps_test_r2, 3),
            "test_mape_pct": round(arps_test_mape * 100, 1),
        },
        "ml_model_role": (
            f"secondary/explanatory only - XGBoost underperformed Arps on this well "
            f"(R2={ml_test_r2:.2f} vs {arps_test_r2:.2f}); used for feature-driver "
            f"context, not as the forecast itself"
        ),
        "top_ml_feature_drivers": top_ml_features,
        "recent_trend": (
            "declining" if merged['oil_rate_bopd'].iloc[-1] < merged['oil_rate_bopd'].iloc[-trend_window_days]
            else "stable/rising"
        ),
    }
    return context


if __name__ == "__main__":
    ctx = build_insight_context(
        cleaned_csv_path="volve_f14_cleaned.csv",
        arps_test_preds_path="arps_baseline_test_preds.csv",
        arps_params={"qi": 568.99, "di": 0.00182},
        arps_test_r2=0.864,
        arps_test_mape=0.098,
        ml_test_r2=-0.061,
        top_ml_features=["yesterday's rate (dominant)", "7-day rolling average rate", "downhole temperature"],
    )
    print(json.dumps(ctx, indent=2))
