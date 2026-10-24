"""Train and compare models for malicious URL classification.

Pipeline
--------
1. Load train/val/test splits (tab-separated ``label<TAB>url``).
2. Sample for laptop-friendly training and extract features with ``features.py``.
3. Train Logistic Regression, SVM (RBF), Random Forest and XGBoost.
4. Run 3-fold cross-validation on a subset (stability check).
5. Select the best model by validation F1 and tune the decision threshold.
6. Evaluate on the official test split and save ``model.pkl`` + ``metrics.json``.

The saved bundle contains everything prediction needs::

    {
        "model": <estimator>,
        "features": [...],           # canonical column order
        "scaler": <StandardScaler or None>,
        "threshold": 0.42,           # F1-optimal
        "high_recall_threshold": 0.30,
        "model_name": "XGBoost",
        "metadata": {...},
    }
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from features import ALL_COLS, extract_all

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dephides" / "dataset" / "small_dataset"
rs = 42


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_split(path: Path) -> pd.DataFrame:
    """Read a ``label<TAB>url`` file, tolerating blank/space-separated lines."""
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split()
            if len(parts) != 2:
                continue
            rows.append(parts)
    return pd.DataFrame(rows, columns=["label", "url"])


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.DataFrame([extract_all(u) for u in df["url"]])[ALL_COLS]
    y = (df["label"] == "phishing").astype(int).to_numpy()
    return X, y


def sample_split(df: pd.DataFrame, n: int | None, seed: int = rs) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def score(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_negative_rate": round(float(fn / (fn + tp)) if (fn + tp) else 0.0, 4),
    }


def tune_thresholds(y_true, proba) -> tuple[float, dict, float, dict]:
    """Return (best_f1_threshold, best_f1_metrics, high_recall_threshold, metrics)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scored = [(t, score(y_true, proba, t)) for t in grid]

    best_t, best_m = max(scored, key=lambda x: (x[1]["f1"], x[1]["accuracy"]))

    # Highest threshold that still achieves >= 0.95 recall (more precise than lower ones).
    high_candidates = [(t, m) for t, m in scored if m["recall"] >= 0.95]
    if high_candidates:
        high_t, high_m = max(high_candidates, key=lambda x: (x[1]["precision"], x[1]["f1"]))
    else:
        high_t, high_m = best_t, best_m
    return best_t, best_m, high_t, high_m


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def make_models() -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", n_jobs=-1
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True, class_weight="balanced", random_state=rs
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=rs, n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, random_state=rs,
            eval_metric="logloss", n_jobs=-1, tree_method="hist",
        ),
    }


SCALED_MODELS = {"LogisticRegression", "SVM (RBF)"}


def cross_validate(model, X, y, folds: int = 3) -> dict:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rs)
    f1s, aucs = [], []
    for tr, te in skf.split(X, y):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        if isinstance(model, (LogisticRegression, SVC)):
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)
        m = clone_model(model)
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        f1s.append(f1_score(yte, p >= 0.5))
        aucs.append(roc_auc_score(yte, p))
    return {
        "cv_f1_mean": round(float(np.mean(f1s)), 4),
        "cv_f1_std": round(float(np.std(f1s)), 4),
        "cv_auc_mean": round(float(np.mean(aucs)), 4),
        "cv_auc_std": round(float(np.std(aucs)), 4),
    }


def clone_model(model):
    from sklearn.base import clone
    return clone(model)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=50_000,
                    help="rows sampled from train.txt (0 = all)")
    ap.add_argument("--val-size", type=int, default=15_000,
                    help="rows sampled from val.txt (0 = all)")
    ap.add_argument("--test-size", type=int, default=20_000,
                    help="rows sampled from test.txt (0 = all)")
    ap.add_argument("--svm-size", type=int, default=12_000,
                    help="SVM is trained on this many rows for speed")
    ap.add_argument("--cv-size", type=int, default=12_000,
                    help="rows used for 3-fold cross-validation")
    ap.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    args = ap.parse_args()

    t0 = time.time()

    def n_or_none(n: int):
        return None if n in (0, -1) else n

    print("Loading data ...")
    train_df = sample_split(load_split(DATA / "train.txt"), n_or_none(args.train_size))
    val_df = sample_split(load_split(DATA / "val.txt"), n_or_none(args.val_size))
    test_df = sample_split(load_split(DATA / "test.txt"), n_or_none(args.test_size))
    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    print("  class balance (train):",
          (train_df["label"] == "phishing").mean().round(4))

    print("Extracting features ...")
    Xtr, ytr = build_xy(train_df)
    Xval, yval = build_xy(val_df)
    Xte, yte = build_xy(test_df)
    print(f"  feature matrix: {Xtr.shape}")

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xval_s = scaler.transform(Xval)
    Xte_s = scaler.transform(Xte)

    models = make_models()
    trained: dict = {}
    val_probas: dict = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        start = time.time()
        # SVM is expensive on large data: train on a subsample.
        if name == "SVM (RBF)":
            idx = np.random.RandomState(rs).choice(
                len(Xtr), size=min(args.svm_size, len(Xtr)), replace=False
            )
            m = clone_model(model).fit(Xtr_s[idx], ytr[idx])
            val_p = m.predict_proba(Xval_s)[:, 1]
        elif name in SCALED_MODELS:
            m = clone_model(model).fit(Xtr_s, ytr)
            val_p = m.predict_proba(Xval_s)[:, 1]
        else:
            m = clone_model(model).fit(Xtr, ytr)
            val_p = m.predict_proba(Xval)[:, 1]
        trained[name] = m
        val_probas[name] = val_p
        m_val = score(yval, val_p, 0.5)
        print(f"  val @0.5 -> acc={m_val['accuracy']} prec={m_val['precision']} "
              f"rec={m_val['recall']} f1={m_val['f1']} auc={m_val['roc_auc']} "
              f"({time.time() - start:.1f}s)")

    # ---- cross-validation (stability) -------------------------------------
    cv_results: dict = {}
    if not args.no_cv:
        print("\n3-fold cross-validation (subset) ...")
        idx = np.random.RandomState(rs).choice(
            len(Xtr), size=min(args.cv_size, len(Xtr)), replace=False
        )
        Xcv, ycv = Xtr.iloc[idx].reset_index(drop=True), ytr[idx]
        for name in ("LogisticRegression", "RandomForest", "XGBoost"):
            print(f"  CV {name} ...")
            cv_results[name] = cross_validate(models[name], Xcv, ycv, folds=3)
            print("   ", cv_results[name])
        cv_results["SVM (RBF)"] = {"note": "skipped (too slow); trained on subsample"}

    # ---- select best model by validation F1 ------------------------------
    print("\nSelecting best model by validation F1 ...")
    val_scores = {n: score(yval, p, 0.5) for n, p in val_probas.items()}
    best_name = max(val_scores, key=lambda n: val_scores[n]["f1"])
    best_model = trained[best_name]
    best_val_p = val_probas[best_name]
    print(f"  best = {best_name} (val F1={val_scores[best_name]['f1']})")

    # ---- threshold tuning on validation ----------------------------------
    best_t, best_m, high_t, high_m = tune_thresholds(yval, best_val_p)
    print(f"  tuned threshold (max F1) = {best_t} -> {best_m}")
    print(f"  high-recall threshold    = {high_t} -> {high_m}")

    # ---- final evaluation on official test split -------------------------
    if best_name in SCALED_MODELS:
        test_p = best_model.predict_proba(Xte_s)[:, 1]
    else:
        test_p = best_model.predict_proba(Xte)[:, 1]

    test_default = score(yte, test_p, 0.5)
    test_tuned = score(yte, test_p, best_t)
    test_high = score(yte, test_p, high_t)

    print("\nTest set @0.5      :", {k: test_default[k] for k in
                                    ("accuracy", "precision", "recall", "f1", "roc_auc")})
    print(f"Test set @{best_t:<5}:", {k: test_tuned[k] for k in
                                    ("accuracy", "precision", "recall", "f1", "roc_auc")})
    print(f"Test set @{high_t:<5}:", {k: test_high[k] for k in
                                    ("accuracy", "precision", "recall", "f1", "roc_auc")})

    # ---- persist ---------------------------------------------------------
    bundle = {
        "model": best_model,
        "features": ALL_COLS,
        "scaler": scaler if best_name in SCALED_MODELS else None,
        "threshold": float(best_t),
        "high_recall_threshold": float(high_t),
        "model_name": best_name,
        "metadata": {
            "n_features": len(ALL_COLS),
            "train_rows": int(len(Xtr)),
            "val_rows": int(len(Xval)),
            "test_rows": int(len(Xte)),
            "val_scores": val_scores,
            "cv": cv_results,
            "test_default": test_default,
            "test_tuned": test_tuned,
            "test_high_recall": test_high,
        },
    }
    joblib.dump(bundle, ROOT / "model.pkl")
    (ROOT / "metrics.json").write_text(
        json.dumps(bundle["metadata"], indent=2), encoding="utf-8"
    )
    print(f"\nSaved model.pkl and metrics.json (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
