"""Zero-dependency web UI for the malicious URL classifier.

Uses only the Python standard library, so it runs anywhere:

    python3 web_app.py            # then open http://localhost:8000

A Streamlit version is also available (`app.py`) if Streamlit is installed.
"""
from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import joblib
import pandas as pd

from features import extract_all

BUNDLE = joblib.load("model.pkl")

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Malicious URL Classifier</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, sans-serif; max-width: 820px; margin: 40px auto;
         padding: 0 20px; line-height: 1.5; }}
  h1 {{ margin-bottom: 0; }}
  .sub {{ color: #888; margin-top: 4px; }}
  form {{ display: flex; gap: 8px; margin: 24px 0; flex-wrap: wrap; }}
  input[type=text] {{ flex: 1; min-width: 280px; padding: 10px; font-size: 16px;
                      border: 1px solid #999; border-radius: 8px; }}
  button {{ padding: 10px 18px; font-size: 16px; border: 0; border-radius: 8px;
            background: #2b7bba; color: white; cursor: pointer; }}
  .result {{ padding: 18px; border-radius: 12px; margin: 18px 0; }}
  .phishing {{ background: #ffe3e3; color: #8a1010; border: 1px solid #e0a0a0; }}
  .safe {{ background: #e3f6e3; color: #0d5a0d; border: 1px solid #a0d0a0; }}
  .big {{ font-size: 24px; font-weight: 700; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 14px; }}
  td, th {{ padding: 6px 10px; border-bottom: 1px solid #ddd; text-align: left; }}
  .bar {{ height: 10px; background: #eee; border-radius: 5px; overflow: hidden; }}
  .bar > span {{ display: block; height: 100%; background: #2b7bba; }}
  details {{ margin-top: 20px; }}
  code {{ font-size: 13px; }}
</style>
</head>
<body>
<h1>🔗 Malicious URL Classifier</h1>
<p class="sub">Model: <b>{model}</b> · {nfeat} features · threshold {threshold:.2f}</p>
<form method="post" action="/predict">
  <input type="text" name="url" placeholder="https://example.com/login"
         value="{url}" autofocus required>
  <button type="submit">Classify</button>
</form>
{result}
<details>
  <summary>Extracted features</summary>
  {features}
</details>
<hr>
<p class="sub">Try: <code>https://www.google.com</code> ·
<code>http://paypal-secure-login.example.com/verify</code> ·
<code>https://secure-bank-update.free-bonus.click/verify?user=abc</code></p>
</body>
</html>"""


def classify(url: str):
    feats = pd.DataFrame([extract_all(url)])[BUNDLE["features"]]
    if BUNDLE.get("scaler") is not None:
        feats_scaled = BUNDLE["scaler"].transform(feats)
    else:
        feats_scaled = feats
    p = float(BUNDLE["model"].predict_proba(feats_scaled)[0][1])
    threshold = float(BUNDLE.get("threshold", 0.5))
    return p, threshold, feats


def result_html(url: str) -> str:
    if not url.strip():
        return ""
    p, threshold, feats = classify(url)
    phishing = p >= threshold
    conf = p if phishing else 1 - p
    cls = "phishing" if phishing else "safe"
    verdict = "PHISHING ⚠️" if phishing else "LEGITIMATE ✅"
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
        for k, v in feats.iloc[0].items()
    )
    return f"""
<div class="result {cls}">
  <div class="big">{verdict}</div>
  <div>Confidence: <b>{conf:.2%}</b> · P(phishing): {p:.2%} · threshold {threshold:.2f}</div>
  <div class="bar" style="margin-top:10px"><span style="width:{p*100:.1f}%"></span></div>
</div>
<details><summary>Extracted features</summary>
<table><tr><th>feature</th><th>value</th></tr>{rows}</table>
</details>"""


def render(url: str = "") -> str:
    return PAGE.format(
        model=BUNDLE.get("model_name", "unknown"),
        nfeat=len(BUNDLE["features"]),
        threshold=float(BUNDLE.get("threshold", 0.5)),
        url=html.escape(url, quote=True),
        result=result_html(url),
        features="",
    )


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: str, status: int = 200):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(render())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        url = parse_qs(body).get("url", [""])[0]
        self._send(render(url))

    def log_message(self, *args):  # quieter console
        pass


def main() -> None:
    host, port = "0.0.0.0", 8000
    print(f"Serving on http://localhost:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
