"""Shared prediction logic used by the CLI and both web interfaces.

Keeping this in one module guarantees the CLI and the web UI use exactly the
same feature order, scaler and decision threshold.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from features import explain_reasons, extract_all, normalize_url

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "model.pkl"


def load_bundle(path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """Load the trained model bundle (model, features, scaler, thresholds)."""
    return joblib.load(path)


def classify(
    url: str,
    bundle: dict,
    threshold: float | None = None,
    high_recall: bool = False,
) -> dict:
    """Classify one URL and return a result dictionary.

    Raises ``ValueError`` for empty / malformed URLs (via ``normalize_url``).
    """
    clean = normalize_url(url)

    feats = pd.DataFrame([extract_all(clean)])[bundle["features"]]
    if bundle.get("scaler") is not None:
        model_input = bundle["scaler"].transform(feats)
    else:
        model_input = feats

    p_phishing = float(bundle["model"].predict_proba(model_input)[0][1])

    if threshold is None:
        threshold = float(
            bundle.get("high_recall_threshold")
            if high_recall
            else bundle.get("threshold", 0.5)
        )
    threshold = float(threshold)

    is_phishing = p_phishing >= threshold
    confidence = p_phishing if is_phishing else 1.0 - p_phishing

    return {
        "url": clean,
        "label": "PHISHING (malicious)" if is_phishing else "LEGITIMATE (safe)",
        "is_phishing": bool(is_phishing),
        "p_phishing": p_phishing,
        "confidence": confidence,
        "threshold": threshold,
        "features": feats,
        "reasons": explain_reasons(clean, is_phishing),
        "model_name": bundle.get("model_name", "unknown"),
    }
