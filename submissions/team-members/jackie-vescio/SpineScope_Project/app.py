# app.py — SpineScope (display-only)

# --- Imports -----------------------------------------------------------------
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --- Page shell ---------------------------------------------------------------
st.set_page_config(page_title="SpineScope", page_icon="🦴", layout="centered")
st.title("🦴 SpineScope")

# --- Sidebar -----------------------------------------------------------------
# st.sidebar.header("Controls")
# view_mode = st.sidebar.radio("Section", ["Model view", "Leaderboard"], index=0)

view_mode = st.sidebar.radio(
    label="",
    options=["Model view", "Leaderboard"],
    index=0,
)

# Show the model picker only when we're in Model view
if view_mode == "Model view":
    model_choice = st.sidebar.selectbox(
        "Pick a model to view",
        [
            "SVC (Tuned)",
            "KNN (Tuned)",
            "SVC (Baseline)",
            "CatBoost (Tuned)",
            "LightGBM (Tuned)",
            "KNN (Baseline)",
            "CatBoost (Baseline)",
            "Logistic Regression (Baseline)",
            "Logistic Regression (Tuned)",
            "LightGBM (Baseline)",
            "Random Forest (Baseline)",
            "XGBoost (Tuned)",
            "XGBoost (Baseline)",
            "Random Forest (Tuned)",
        ],
        index=0,
    )

use_mlflow = st.sidebar.checkbox("Use live MLflow metrics", value=True)

# --- Snapshot metrics (from your notebook) -----------------------------------
LB = {
    "SVC (Tuned)":                   dict(acc=0.9032, mic=0.9032, mac=0.8893, wgt=0.9032),
    "KNN (Tuned)":                   dict(acc=0.8710, mic=0.8710, mac=0.8591, wgt=0.8736),
    "SVC (Baseline)":                dict(acc=0.8548, mic=0.8548, mac=0.8398, wgt=0.8572),
    "CatBoost (Tuned)":              dict(acc=0.8387, mic=0.8387, mac=0.8239, wgt=0.8420),
    "LightGBM (Tuned)":              dict(acc=0.8387, mic=0.8387, mac=0.8200, wgt=0.8406),
    "KNN (Baseline)":                dict(acc=0.8226, mic=0.8226, mac=0.8082, wgt=0.8268),
    "CatBoost (Baseline)":           dict(acc=0.8387, mic=0.8387, mac=0.8200, wgt=0.8406),
    "Logistic Regression (Baseline)":dict(acc=0.8226, mic=0.8226, mac=0.7943, wgt=0.8213),
    "Logistic Regression (Tuned)":   dict(acc=0.8226, mic=0.8226, mac=0.7943, wgt=0.8213),
    "LightGBM (Baseline)":           dict(acc=0.8065, mic=0.8065, mac=0.7840, wgt=0.8087),
    "Random Forest (Baseline)":      dict(acc=0.8065, mic=0.8065, mac=0.7786, wgt=0.8065),
    "XGBoost (Tuned)":               dict(acc=0.8065, mic=0.8065, mac=0.7786, wgt=0.8065),
    "XGBoost (Baseline)":            dict(acc=0.7903, mic=0.7903, mac=0.7632, wgt=0.7916),
    "Random Forest (Tuned)":         dict(acc=0.7903, mic=0.7903, mac=0.7632, wgt=0.7916),
}

# --- Quiet MLflow helpers (optional; never error the UI) ---------------------
def _mlruns_uri():
    return (Path(__file__).resolve().parent / "mlruns").absolute().as_uri()

def _latest_metrics_from_mlflow(run_name: str):
    """Return metrics dict or None."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(_mlruns_uri())
        client = MlflowClient()
        exp = client.get_experiment_by_name("SpineScope")
        if exp is None:
            return None

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"attributes.run_name = '{run_name}' and attributes.status = 'FINISHED'",
            order_by=["attribute.start_time DESC"],
            max_results=1,
        )
        if not runs:
            return None

        m = runs[0].data.metrics
        want = ("accuracy", "f1_micro", "f1_macro", "f1_weighted")
        if not all(k in m for k in want):
            return None

        return dict(acc=m["accuracy"], mic=m["f1_micro"], mac=m["f1_macro"], wgt=m["f1_weighted"])
    except Exception:
        return None

def _latest_cm_from_mlflow(run_name: str):
    """Return 2x2 numpy array or None by reading a CM artifact."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(_mlruns_uri())
        client = MlflowClient()
        exp = client.get_experiment_by_name("SpineScope")
        if exp is None:
            return None

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"attributes.run_name = '{run_name}' and attributes.status = 'FINISHED'",
            order_by=["attribute.start_time DESC"],
            max_results=1,
        )
        if not runs:
            return None

        run_id = runs[0].info.run_id
        names = ["cm.json", "confusion_matrix.json", "cm.csv", "confusion_matrix.csv"]

        root = client.list_artifacts(run_id, "")
        candidates = {a.path for a in root}
        target = next((n for n in names if n in candidates), None)
        if target is None:
            for a in root:
                if a.is_dir:
                    inner = client.list_artifacts(run_id, a.path)
                    inner_paths = {f"{a.path}/{b.path}" for b in inner}
                    target = next((p for n in names for p in inner_paths if p.endswith(n)), None)
                    if target:
                        break
        if target is None:
            return None

        with tempfile.TemporaryDirectory() as tmp:
            local = client.download_artifacts(run_id, target, tmp)
            if local.endswith(".json"):
                with open(local, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                arr = obj.get("confusion_matrix", obj)
                cm = np.asarray(arr, dtype=int)
            else:
                cm = np.loadtxt(local, delimiter=",", dtype=int)
        return cm if cm.shape == (2, 2) else None
    except Exception:
        return None

# =========================
#   VIEW: MODEL VIEW
# =========================
if view_mode == "Model view":
    # --- Metrics header (live if available; else snapshot) -------------------
    row = _latest_metrics_from_mlflow(model_choice) if use_mlflow else None
    row = row or LB.get(model_choice)

    st.markdown(f"### {model_choice}")
    c1, c2, c3, c4 = st.columns(4)
    if row:
        c1.metric("Accuracy",      f"{row['acc']:.4f}")
        c2.metric("F1 (micro)",    f"{row['mic']:.4f}")
        c3.metric("F1 (macro)",    f"{row['mac']:.4f}")
        c4.metric("F1 (weighted)", f"{row['wgt']:.4f}")
    else:
        st.info("No metrics available.")

    # --- Confusion matrix (MLflow if present; else snapshot) -----------------
    SNAPSHOT_CM = {
        "XGBoost (Baseline)":               [[35, 7], [6, 14]],
        "XGBoost (Tuned)":                  [[36, 6], [6, 14]],
        "KNN (Baseline)":                   [[34, 8], [3, 17]],
        "KNN (Tuned)":                      [[36, 6], [2, 18]],
        "LightGBM (Baseline)":              [[35, 7], [5, 15]],
        "LightGBM (Tuned)":                 [[36, 6], [4, 16]],
        "CatBoost (Baseline)":              [[36, 6], [4, 16]],
        "CatBoost (Tuned)":                 [[35, 7], [4, 16]],
        "SVC (Baseline)":                   [[36, 6], [3, 17]],
        "SVC (Tuned)":                      [[39, 3], [3, 17]],
        "Logistic Regression (Baseline)":   [[37, 7], [6, 14]],
        "Logistic Regression (Tuned)":      [[37, 5], [6, 14]],
        "Random Forest (Baseline)":         [[36, 6], [6, 14]],
        "Random Forest (Tuned)":            [[35, 7], [6, 14]],
    }

    st.markdown("#### Confusion matrix")

    cm = _latest_cm_from_mlflow(model_choice) if use_mlflow else None
    if cm is None and model_choice in SNAPSHOT_CM:
        cm = np.array(SNAPSHOT_CM[model_choice], dtype=int)

    if cm is not None:
        fig, ax = plt.subplots(figsize=(4.8, 4.3))
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{model_choice} — Confusion Matrix")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Abnormal", "Normal"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Abnormal", "Normal"])

        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Count", rotation=270, labelpad=12)
        st.pyplot(fig, clear_figure=True)
    else:
        st.caption("No confusion matrix available for this model.")

# =========================
#   VIEW: LEADERBOARD
# =========================
else:
    st.markdown("### Leaderboard — Best Model Performance in Ascending Order")

    ORDER = [
        "SVC (Baseline)",
        "SVC (Tuned)",
        "KNN (Tuned)",
        "CatBoost (Tuned)",
        "LightGBM (Tuned)",
        "KNN (Baseline)",
        "CatBoost (Baseline)",
        "Logistic Regression (Baseline)",
        "Logistic Regression (Tuned)",
        "LightGBM (Baseline)",
        "Random Forest (Baseline)",
        "Random Forest (Tuned)",
        "XGBoost (Tuned)",
        "XGBoost (Baseline)",
    ]

    rows = []
    for name in ORDER:
        m = _latest_metrics_from_mlflow(name) if use_mlflow else None
        m = m or LB.get(name)
        if m:
            rows.append({
                "Model": name,
                "Accuracy":      f"{m['acc']:.4f}",
                "F1 (micro)":    f"{m['mic']:.4f}",
                "F1 (macro)":    f"{m['mac']:.4f}",
                "F1 (weighted)": f"{m['wgt']:.4f}",
            })

    lb_df = pd.DataFrame(rows)
    st.dataframe(lb_df, hide_index=True, use_container_width=True)
