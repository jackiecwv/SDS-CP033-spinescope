# app.py — SpineScope (with tie flags + MLflow last updated timestamp)

# --- Imports -----------------------------------------------------------------
import json
import tempfile
from functools import lru_cache
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --- Page shell ---------------------------------------------------------------
st.set_page_config(page_title="SpineScope", page_icon="🩻", layout="centered")
st.title("🩻 SpineScope")

# --- Sidebar -----------------------------------------------------------------
view_mode = st.sidebar.radio(
    label="",
    options=["Leaderboard", "Model view"],
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

MODEL_ORDER = [
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

# --- Quiet MLflow helpers (never error the UI) --------------------------------
def _mlruns_uri():
    return (Path(__file__).resolve().parent / "mlruns").absolute().as_uri()

@lru_cache(maxsize=256)
def _latest_metrics_from_mlflow(run_name: str):
    """Return (metrics dict, last_updated datetime) or (None, None)."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(_mlruns_uri())
        client = MlflowClient()
        exp = client.get_experiment_by_name("SpineScope")
        if exp is None:
            return None, None

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"attributes.run_name = '{run_name}' and attributes.status = 'FINISHED'",
            order_by=["attribute.start_time DESC"],
            max_results=1,
        )
        if not runs:
            return None, None

        r = runs[0]
        m = r.data.metrics
        want = ("accuracy", "f1_micro", "f1_macro", "f1_weighted")
        if not all(k in m for k in want):
            return None, None

        last_updated = datetime.fromtimestamp(r.info.end_time / 1000.0)
        return dict(acc=m["accuracy"], mic=m["f1_micro"], mac=m["f1_macro"], wgt=m["f1_weighted"]), last_updated
    except Exception:
        return None, None

@lru_cache(maxsize=256)
def _latest_cm_from_mlflow(run_name: str):
    """Return 2x2 numpy array or None by reading a CM artifact (cached)."""
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

# --- Snapshot CMs (fallbacks) -------------------------------------------------
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

# =========================
#   VIEW: MODEL VIEW
# =========================
if view_mode == "Model view":
    model_choice = st.sidebar.selectbox(
        "Pick a model to view",
        MODEL_ORDER,
        index=0,
    )

    row, last_updated = _latest_metrics_from_mlflow(model_choice) if use_mlflow else (None, None)
    row = row or LB.get(model_choice)

    st.markdown(f"### {model_choice}")
    c1, c2, c3, c4 = st.columns(4)
    if row:
        c1.metric("Accuracy",      f"{row['acc']:.4f}")
        c2.metric("F1 (micro)",    f"{row['mic']:.4f}")
        c3.metric("F1 (macro)",    f"{row['mac']:.4f}")
        c4.metric("F1 (weighted)", f"{row['wgt']:.4f}")
        if last_updated:
            st.caption(f"Last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("No metrics available.")

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
    st.markdown("### Leaderboard \ncurrent models (test metrics)")

    rank_metric = st.selectbox(
        "Rank by",
        options=[
            ("F1 (weighted)", "wgt"),
            ("Accuracy", "acc"),
            ("F1 (micro)", "mic"),
            ("F1 (macro)", "mac"),
        ],
        index=0,
        format_func=lambda x: x[0],
    )[1]

    rows = []
    last_update_times = []
    for name in MODEL_ORDER:
        m, t = _latest_metrics_from_mlflow(name) if use_mlflow else (None, None)
        m = m or LB.get(name)
        if m:
            rows.append({
                "Model":        name,
                "Accuracy":     float(m["acc"]),
                "F1_micro":     float(m["mic"]),
                "F1_macro":     float(m["mac"]),
                "F1_weighted":  float(m["wgt"]),
                "_rank_val":    float(m[rank_metric]),
            })
            if t:
                last_update_times.append(t)

    if rows:
        lb = pd.DataFrame(rows)

        lb["_rank_key"] = lb["_rank_val"].round(6)
        lb["Tied on rank metric?"] = lb.duplicated("_rank_key", keep=False).map({True: "✓", False: ""})

        lb = lb.sort_values(
            by=["_rank_val", "F1_macro", "Accuracy"],
            ascending=[False, False, False],
            ignore_index=True,
        )
        lb.insert(0, "Rank", lb.index + 1)

        show = lb[["Rank", "Model", "Tied on rank metric?", "F1_weighted", "Accuracy", "F1_micro", "F1_macro"]].copy()
        show.rename(columns={
            "F1_weighted": "F1 (weighted)  🏆",
            "F1_micro":    "F1 (micro)",
            "F1_macro":    "F1 (macro)"
        }, inplace=True)

        for col in ["F1 (weighted)  🏆", "Accuracy", "F1 (micro)", "F1 (macro)"]:
            show[col] = show[col].map(lambda x: f"{x:.4f}")

        st.caption("Ranked by **{}**; toggle live MLflow to refresh.".format(
            "F1 (weighted)" if rank_metric == "wgt" else
            "Accuracy" if rank_metric == "acc" else
            "F1 (micro)" if rank_metric == "mic" else
            "F1 (macro)"
        ))
        if use_mlflow and last_update_times:
            st.caption(f"Last updated: {max(last_update_times).strftime('%Y-%m-%d %H:%M:%S')}")

        st.dataframe(show, hide_index=True, use_container_width=True)

        with st.expander("What these metrics tell you"):
            st.markdown("""
- **Accuracy / F1-micro**: identical for single-label classification; overall % correct.
- **F1-macro**: unweighted mean across classes; highlights minority-class issues.
- **F1-weighted**: macro F1 weighted by class support; balances class quality with class size.
            """)
    else:
        st.info("No metrics available.")

# --- End of app.py -----------------------------------------------------------
