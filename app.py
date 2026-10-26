"""Streamlit demo UI for the malicious URL classifier.

Run with:
    streamlit run app.py

The UI reuses ``classifier.py`` and the trained ``model.pkl`` bundle, so the
same logic as ``predict.py`` is used. It also shows the model's global feature
importances to make predictions more explainable.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from classifier import DEFAULT_MODEL_PATH, classify, load_bundle

st.set_page_config(page_title="Malicious URL Classifier", page_icon="🔗", layout="wide")


@st.cache_resource
def get_bundle():
    return load_bundle(DEFAULT_MODEL_PATH)


@st.cache_data
def get_importances(features: tuple[str, ...]) -> pd.Series:
    bundle = get_bundle()
    model = bundle["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = abs(model.coef_[0])
    else:  # pragma: no cover
        values = [0.0] * len(features)
    return pd.Series(values, index=list(features)).sort_values(ascending=False)


def main() -> None:
    try:
        bundle = get_bundle()
    except FileNotFoundError:
        st.error("model.pkl not found. Run `python3 train_model.py` first.")
        return

    default_threshold = float(bundle.get("threshold", 0.5))

    st.title("🔗 Malicious URL Classifier")
    st.caption(
        f"Model: **{bundle.get('model_name', 'unknown')}** · "
        f"{len(bundle['features'])} features · "
        f"decision threshold tuned on validation data"
    )

    with st.sidebar:
        st.header("Settings")
        threshold = st.slider(
            "Decision threshold",
            min_value=0.05, max_value=0.95,
            value=round(default_threshold, 2), step=0.01,
            help="Lower = catch more phishing (higher recall) but more false alarms.",
        )
        st.caption(
            f"Saved F1-optimal threshold: **{default_threshold:.2f}**  \n"
            f"High-recall threshold: **{bundle.get('high_recall_threshold', default_threshold):.2f}**"
        )

        st.divider()
        st.subheader("Try an example")
        examples = {
            "Legit: Google": "https://www.google.com",
            "Legit (scary-looking): PayPal sign-in":
                "https://www.paypal.com/signin?returnUri=https://www.paypal.com/myaccount/summary",
            "Phishing: fake PayPal":
                "http://paypal-secure-login.example.com/verify/account",
            "Phishing: IP host": "http://192.168.1.10/login.php",
            "Phishing: suspicious TLD":
                "https://secure-bank-update.free-bonus.click/verify?user=abc",
        }
        chosen = st.radio("Load example", list(examples), index=0)
        if st.button("Use this example"):
            st.session_state["url"] = examples[chosen]

    url = st.text_input(
        "Enter a URL", value=st.session_state.get("url", ""),
        placeholder="https://example.com/login",
    )

    if st.button("🔍 Classify", type="primary") and url.strip():
        try:
            result = classify(url, bundle, threshold=threshold)
        except ValueError as exc:
            st.error(f"Invalid URL: {exc}")
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Verdict", "PHISHING ⚠️" if result["is_phishing"] else "LEGITIMATE ✅")
        col2.metric("Confidence", f"{result['confidence']:.2%}")
        col3.metric("P(phishing)", f"{result['p_phishing']:.2%}")

        st.progress(min(max(result["p_phishing"], 0.0), 1.0))
        if result["is_phishing"]:
            st.error("This URL is classified as malicious (phishing).")
        else:
            st.success("This URL is classified as legitimate (safe).")

        st.divider()
        left, right = st.columns(2)

        with left:
            st.subheader("Extracted features")
            st.dataframe(
                result["features"].T.rename(columns={0: "value"}),
                use_container_width=True, height=420,
            )

        with right:
            st.subheader("Top global feature importances")
            importances = get_importances(tuple(bundle["features"])).head(12)
            st.bar_chart(importances)
            st.caption("Global importance from the trained model (not per-URL attribution).")

    elif not url.strip():
        st.info("Enter a URL above, or pick an example from the sidebar.")

    st.divider()
    st.caption(
        "Tip: lower the threshold to reduce missed phishing (false negatives) at the "
        "cost of more false alarms (false positives)."
    )


if __name__ == "__main__":
    main()
