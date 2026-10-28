"""Predict whether a URL is phishing (malicious) or legitimate.

Usage
-----
    python3 predict.py "https://example.com"     # single URL
    python3 predict.py --loop                    # interactive loop
    python3 predict.py                           # prompt once

The decision threshold is loaded from ``model.pkl`` (tuned on the validation
split), not hard-coded to 0.5. ``--high-recall`` uses the higher-sensitivity
threshold to reduce missed phishing at the cost of more false alarms.
"""
from __future__ import annotations

import argparse

from classifier import DEFAULT_MODEL_PATH, classify, load_bundle


def print_result(result: dict) -> None:
    print(f"URL        : {result['url']}")
    print(f"Result     : {result['label']}")
    print(f"Confidence : {result['confidence']:.2%}")
    print(f"P(phishing): {result['p_phishing']:.2%}  "
          f"(threshold: {result['threshold']:.2f})")
    print("Why:")
    for reason in result["reasons"]:
        print(f"  - {reason}")
    print()


def run_one(url: str, bundle: dict, high_recall: bool) -> bool:
    """Classify one URL, returning False if the input was invalid."""
    try:
        print_result(classify(url, bundle, high_recall=high_recall))
        return True
    except ValueError as exc:
        print(f"Invalid URL: {exc}\n")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Malicious URL classifier")
    ap.add_argument("url", nargs="*", help="URL to classify (omit for prompt)")
    ap.add_argument("--loop", action="store_true", help="keep asking for URLs")
    ap.add_argument("--high-recall", action="store_true",
                    help="use the higher-recall threshold")
    ap.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="path to model bundle")
    args = ap.parse_args()

    try:
        bundle = load_bundle(args.model)
    except FileNotFoundError:
        raise SystemExit(
            f"Model not found at {args.model}. Run `python3 train_model.py` first."
        )

    print(f"Model: {bundle.get('model_name', 'unknown')}  "
          f"({len(bundle['features'])} features, threshold={bundle.get('threshold', 0.5):.2f})\n")

    if args.url:
        run_one(" ".join(args.url), bundle, args.high_recall)
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
        run_one(url, bundle, args.high_recall)
        if not args.loop:
            break


if __name__ == "__main__":
    main()
