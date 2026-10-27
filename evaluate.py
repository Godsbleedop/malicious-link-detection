"""Evaluate the saved model and compare feature configurations.

Produces:
    docs/figures/roc_curve.png
    docs/figures/pr_curve.png
    docs/figures/confusion_matrix.png
    docs/figures/feature_importance.png
    docs/figures/model_comparison.png
    metrics_configs.json   (lexical-only / host-only / combined results)

Usage:
    python3 evaluate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import ALL_COLS, HOST_COLS, LEXICAL_COLS, extract_all
from train_model import build_xy, load_split, sample_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dephides" / "dataset" / "small_dataset"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
rs = 42


def get_test(n: int = 20_000):
    df = sample_split(load_split(DATA / "test.txt"), n)
    return build_xy(df)


def plot_saved_model(Xte, yte) -> dict:
    bundle = joblib.load(ROOT / "model.pkl")
    X = Xte[bundle["features"]]
    if bundle.get("scaler") is not None:
        X = bundle["scaler"].transform(X)
    proba = bundle["model"].predict_proba(X)[:, 1]
    threshold = float(bundle.get("threshold", 0.5))
    pred = (proba >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(yte, proba, ax=ax, name=bundle["model_name"])
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("ROC curve — official test split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "roc_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(yte, proba, ax=ax, name=bundle["model_name"])
    ax.set_title("Precision–Recall curve — official test split")
    fig.tight_layout()
    fig.savefig(FIG / "pr_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix(yte, pred), display_labels=["legitimate", "phishing"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix @ threshold {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    model = bundle["model"]
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=bundle["features"]).sort_values()
        top = imp.tail(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        top.plot.barh(ax=ax, color="#2b7bba")
        ax.set_title("Top 15 feature importances")
        ax.set_xlabel("importance")
        fig.tight_layout()
        fig.savefig(FIG / "feature_importance.png", dpi=200)
        plt.close(fig)

    return {
        "model": bundle["model_name"],
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "precision": round(float(precision_score(yte, pred)), 4),
        "recall": round(float(recall_score(yte, pred)), 4),
        "f1": round(float(f1_score(yte, pred)), 4),
        "roc_auc": round(float(roc_auc_score(yte, proba)), 4),
    }


def compare_configs(Xtr, ytr, Xte, yte) -> pd.DataFrame:
    results = []
    for name, cols in {
        "lexical-only": LEXICAL_COLS,
        "host-only": HOST_COLS,
        "combined": ALL_COLS,
    }.items():
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, random_state=rs,
            eval_metric="logloss", n_jobs=-1, tree_method="hist",
        )
        model.fit(Xtr[cols], ytr)
        proba = model.predict_proba(Xte[cols])[:, 1]
        pred = (proba >= 0.5).astype(int)
        results.append({
            "config": name,
            "n_features": len(cols),
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "precision": round(float(precision_score(yte, pred)), 4),
            "recall": round(float(recall_score(yte, pred)), 4),
            "f1": round(float(f1_score(yte, pred)), 4),
            "roc_auc": round(float(roc_auc_score(yte, proba)), 4),
        })
        print(f"  {name:<13} n={len(cols):>2}  {results[-1]}")

    df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    df.set_index("config")[["accuracy", "precision", "recall", "f1"]].plot.bar(ax=axes[0])
    axes[0].set_title("Metrics by feature configuration")
    axes[0].set_ylim(0.8, 1.0)
    axes[0].tick_params(axis="x", rotation=0)
    df.set_index("config")["roc_auc"].plot.bar(ax=axes[1], color="#2b7bba")
    axes[1].set_title("ROC-AUC by feature configuration")
    axes[1].set_ylim(0.9, 1.0)
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIG / "model_comparison.png", dpi=200)
    plt.close(fig)
    return df


def main() -> None:
    print("Loading test split ...")
    Xte, yte = get_test()

    print("Evaluating saved model ...")
    saved = plot_saved_model(Xte, yte)
    print(" ", saved)

    print("Comparing feature configurations ...")
    # Train on a training subset, evaluate on the same test split.
    train_df = sample_split(load_split(DATA / "train.txt"), 20_000)
    Xtr, ytr = build_xy(train_df)
    configs = compare_configs(Xtr, ytr, Xte, yte)

    out = {"saved_model_on_test": saved, "feature_configs": configs.to_dict(orient="records")}
    (ROOT / "metrics_configs.json").write_text(json.dumps(out, indent=2))
    print("Saved metrics_configs.json and figures in docs/figures/")


if __name__ == "__main__":
    main()
