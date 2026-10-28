"""Minimal Streamlit UI for the URL safety checker.

Run with:
    streamlit run app.py

Only the verdict and a confidence percentage are shown; no model, feature or
threshold details are exposed.
"""
from __future__ import annotations

import streamlit as st

from classifier import DEFAULT_MODEL_PATH, classify, load_bundle

st.set_page_config(page_title="URL Checker", layout="centered")

st.markdown(
    """
    <style>
      #MainMenu, footer, header { visibility: hidden; }
      .block-container { max-width: 560px; padding-top: 4rem; }
      h1 { font-size: 1.35rem !important; font-weight: 600 !important; }
      .verdict { font-size: 1.15rem; font-weight: 600; margin: 0.75rem 0 0.25rem; }
      .phishing { color: #b91c1c; }
      .safe { color: #15803d; }
      .meta { color: #6b7280; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_bundle():
    return load_bundle(DEFAULT_MODEL_PATH)


def main() -> None:
    try:
        bundle = get_bundle()
    except FileNotFoundError:
        st.error("Unable to start the classifier.")
        return

    st.title("URL Checker")
    st.caption("Check whether a link is safe or malicious.")

    url = st.text_input(
        "URL", placeholder="https://example.com", label_visibility="collapsed"
    )

    if st.button("Check", type="primary") and url.strip():
        try:
            result = classify(url, bundle)
        except ValueError:
            st.error("Invalid URL.")
            return

        cls = "phishing" if result["is_phishing"] else "safe"
        verdict = "Phishing" if result["is_phishing"] else "Legitimate"
        st.markdown(
            f'<p class="verdict {cls}">{verdict}</p>'
            f'<p class="meta">Confidence: {result["confidence"]:.2%}</p>',
            unsafe_allow_html=True,
        )
        st.progress(min(max(result["p_phishing"], 0.0), 1.0))


if __name__ == "__main__":
    main()
