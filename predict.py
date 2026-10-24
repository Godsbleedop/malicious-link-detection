"""Predict whether a URL is phishing (malicious) or legitimate.

Usage
-----
    python3 predict.py "https://example.com"     # single URL
    python3 predict.py --loop                    # interactive loop
    python3 predict.py                           # prompt once

The decision threshold is loaded from ``model.pkl`` (tuned on the validation
split), not hard-coded to 0.5. A ``--high-recall`` flag uses the higher-sensitivity
threshold to reduce missed phishing at the cost of more false alarms.
"""
from __future__ import annotations

import argparse
import sys

import joblib
import pandas as pd

from features import extract_all  # noqa: F401  (kept for backwards compatibility)

MODEL_PATH = "model.pkl"


def load_bundle(path: str = MODEL_PATH) -> dict:
    return joblib.load(path)


def predict_url(url: str, bundle: dict, high_recall: bool = False):
    """Return (label, probability_of_predicted_class, p_phishing, threshold)."""
    model = bundle["model"]
    feats = pd.DataFrame([extract_all(url)])[bundle["features"]]
    if bundle.get("scaler") is not None:
        feats = bundle["scaler"].transform(feats)

    p_phishing = float(model.predict_proba(feats)[0][1])
    threshold = float(
        bundle.get("high_recall_threshold") if high_recall else bundle.get("threshold", 0.5)
    )
    pred = int(p_phishing >= threshold)
    confidence = p_phishing if pred == 1 else 1.0 - p_phishing
    label = "PHISHING (malicious)" if pred == 1 else "LEGITIMATE (safe)"
    return label, confidence, p_phishing, threshold


def print_result(url: str, result) -> None:
    label, confidence, p_phishing, threshold = result
    print(f"URL        : {url}")
    print(f"Result     : {label}")
    print(f"Confidence : {confidence:.2%}")
    print(f"P(phishing): {p_phishing:.2%}  (threshold: {threshold:.2f})")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Malicious URL classifier")
    ap.add_argument("url", nargs="*", help="URL to classify (omit for prompt)")
    ap.add_argument("--loop", action="store_true", help="keep asking for URLs")
    ap.add_argument("--high-recall", action="store_true",
                    help="use the higher-recall threshold")
    ap.add_argument("--model", default=MODEL_PATH, help="path to model bundle")
    args = ap.parse_args()

    bundle = load_bundle(args.model)
    print(f"Model: {bundle.get('model_name', 'unknown')}  "
          f"({len(bundle['features'])} features, threshold={bundle.get('threshold', 0.5):.2f})\n")

    if args.url:
        url = " ".join(args.url)
        print_result(url, predict_url(url, bundle, args.high_recall))
        return

    while True:
        try:
            url = input("Enter URL (or 'exit' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not url or url.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break
        print_result(url, predict_url(url, bundle, args.high_recall))
        if not args.loop:
            break


if __name__ == "__main__":
    main()
