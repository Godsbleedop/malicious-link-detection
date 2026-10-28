"""Minimal web UI for the URL safety checker.

Only the verdict and a confidence percentage are shown; no model, feature or
threshold details are exposed.

    python3 web_app.py        # then open http://localhost:8000
"""
from __future__ import annotations

import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from classifier import DEFAULT_MODEL_PATH, classify, load_bundle

try:
    BUNDLE = load_bundle(DEFAULT_MODEL_PATH)
except FileNotFoundError:  # pragma: no cover
    BUNDLE = None

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>URL Checker</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; min-height: 100vh; display: flex; align-items: center;
    justify-content: center; background: #f4f5f7; color: #1f2933;
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
    font-size: 15px; line-height: 1.5;
  }}
  .card {{
    width: 100%; max-width: 560px; margin: 24px; padding: 32px;
    background: #ffffff; border: 1px solid #d9dce1; border-radius: 8px;
  }}
  h1 {{ margin: 0 0 6px; font-size: 20px; font-weight: 600; }}
  .sub {{ margin: 0 0 24px; font-size: 13px; color: #6b7280; }}
  form {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  input[type=text] {{
    flex: 1; min-width: 240px; padding: 10px 12px; font-size: 15px;
    font-family: inherit; color: #1f2933; background: #ffffff;
    border: 1px solid #b8bec7; border-radius: 6px; outline: none;
  }}
  input[type=text]:focus {{ border-color: #1d4ed8; }}
  button {{
    padding: 10px 18px; font-size: 15px; font-family: inherit; font-weight: 500;
    color: #ffffff; background: #1d4ed8; border: 0; border-radius: 6px;
    cursor: pointer;
  }}
  button:hover {{ background: #1e40af; }}
  .result {{ margin-top: 24px; padding-top: 20px; border-top: 1px solid #e5e7eb; }}
  .verdict {{ font-size: 17px; font-weight: 600; }}
  .phishing {{ color: #b91c1c; }}
  .safe {{ color: #15803d; }}
  .error {{ color: #b91c1c; font-size: 14px; }}
  .meta {{ margin-top: 4px; font-size: 13px; color: #6b7280;
           font-variant-numeric: tabular-nums; }}
  .bar {{ margin-top: 12px; height: 6px; background: #e5e7eb; border-radius: 3px;
          overflow: hidden; }}
  .bar > span {{ display: block; height: 100%; background: #1d4ed8; }}
</style>
</head>
<body>
<main class="card">
  <h1>URL Checker</h1>
  <p class="sub">Check whether a link is safe or malicious.</p>
  <form method="post" action="/check">
    <input type="text" name="url" placeholder="https://example.com"
           value="{url}" autofocus required>
    <button type="submit">Check</button>
  </form>
  {result}
</main>
</body>
</html>"""


def result_html(url: str) -> str:
    if not url.strip():
        return ""
    try:
        result = classify(url, BUNDLE)
    except ValueError:
        return '<div class="result"><p class="error">Invalid URL.</p></div>'
    except Exception:
        return '<div class="result"><p class="error">Unable to check this URL.</p></div>'

    cls = "phishing" if result["is_phishing"] else "safe"
    verdict = "Phishing" if result["is_phishing"] else "Legitimate"
    conf = result["confidence"]
    return f"""
<div class="result">
  <p class="verdict {cls}">{verdict}</p>
  <p class="meta">Confidence: {conf:.2%}</p>
  <div class="bar"><span style="width:{conf*100:.1f}%"></span></div>
</div>"""


def render(url: str = "") -> str:
    return PAGE.format(
        url=html.escape(url, quote=True),
        result=result_html(url),
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
        body = self.rfile.read(length).decode("utf-8", errors="ignore")
        url = parse_qs(body).get("url", [""])[0]
        self._send(render(url))

    def log_message(self, *args):
        pass


def main() -> None:
    if BUNDLE is None:
        raise SystemExit("model.pkl not found. Run `python3 train_model.py` first.")
    host, port = "0.0.0.0", 8000
    print(f"Serving on http://localhost:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
