"""Streamlit demo UI for the malicious URL classifier.

Run with:
    streamlit run app.py

The UI reuses ``features.py`` and the trained ``model.pkl`` bundle, so the same
logic as ``predict.py`` is used. It also shows the model's global feature
importances to make predictions more explainable.
"""
from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from features import extract_all

st.set_page_config(page_title="Malicious URL Classifier", page_icon="🔗", layout="wide")


@st.cache_resource
def load_bundle(path: str = "model.pkl") -> dict:
    return joblib.load(path)


@st.cache_data
def get_importances(features: tuple[str, ...]) -> pd.Series:
    bundle = load_bundle()
    model = bundle["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = abs(model.coef_[0])
    else:  # pragma: no cover
        values = [0.0] * len(features)
    return pd.Series(values, index=list(features)).sort_values(ascending=False)


def classify(url: str, bundle: dict, threshold: float):
    feats = pd.DataFrame([extract_all(url)])[bundle["features"]]
    if bundle.get("scaler") is not None:
        feats_scaled = bundle["scaler"].transform(feats)
    else:
        feats_scaled = feats
    p_phishing = float(bundle["model"].predict_proba(feats_scaled)[0][1])
    is_phishing = p_phishing >= threshold
    return p_phishing, is_phishing, feats


def main() -> None:
    bundle = load_bundle()
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
        if st.button("Reset to saved threshold"):
            threshold = default_threshold

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
        p_phishing, is_phishing, feats = classify(url.strip(), bundle, threshold)
        confidence = p_phishing if is_phishing else 1 - p_phishing

        col1, col2, col3 = st.columns(3)
        col1.metric("Verdict", "PHISHING ⚠️" if is_phishing else "LEGITIMATE ✅")
        col2.metric("Confidence", f"{confidence:.2%}")
        col3.metric("P(phishing)", f"{p_phishing:.2%}")

        st.progress(min(max(p_phishing, 0.0), 1.0))
        if is_phishing:
            st.error("This URL is classified as malicious (phishing).")
        else:
            st.success("This URL is classified as legitimate (safe).")

        st.divider()
        left, right = st.columns([1, 1])

        with left:
            st.subheader("Extracted features")
            st.dataframe(
                feats.T.rename(columns={0: "value"}),
                use_container_width=True, height=420,
            )

        with right:
            st.subheader("Top global feature importances")
            importances = get_importances(tuple(bundle["features"])).head(12)
            st.bar_chart(importances)
            st.caption("Global importance from the trained model (not per-URL attribution).")

    elif url.strip() == "":
        st.info("Enter a URL above, or pick an example from the sidebar.")

    st.divider()
    st.caption(
        "Tip: lower the threshold to reduce missed phishing (false negatives) at the "
        "cost of more false alarms (false positives)."
    )


if __name__ == "__main__":
    main()
