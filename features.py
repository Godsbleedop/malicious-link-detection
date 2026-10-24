"""Lexical and host-based feature extraction for malicious URL classification.

Feature groups
--------------
LEXICAL_COLS : properties of the raw URL string.
HOST_COLS    : properties of the host / registered domain (via tldextract).

`extract_all(url)` merges both and `ALL_COLS` is the canonical column order used
by training, evaluation and prediction.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

import tldextract

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SUSPICIOUS_WORDS = [
    "login", "verify", "secure", "account", "update", "bank", "password",
    "signin", "confirm", "webscr", "ebayisapi", "paypal", "free", "lucky",
    "bonus", "click", "urgent", "suspend", "validate", "recover", "unlock",
    "billing", "invoice", "payment", "wallet", "support", "alert",
]

# TLDs frequently abused for phishing / cheap throw-away domains.
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "click", "xyz", "top", "work", "gdn",
    "loan", "download", "review", "country", "kim", "cricket", "science",
    "party", "racing", "win", "bid", "stream", "men", "date", "faith",
    "zip", "mov", "rest", "monster", "quest", "cyou", "sbs", "cfd",
}

# Brands commonly impersonated in subdomains / paths.
BRANDS = [
    "paypal", "google", "apple", "amazon", "facebook", "microsoft",
    "netflix", "ebay", "instagram", "whatsapp", "linkedin", "dropbox",
    "adobe", "coinbase", "binance", "chase", "wellsfargo", "hsbc",
    "bankofamerica", "outlook", "icloud", "yahoo", "twitter", "steam",
]

# Common web file extensions.
FILE_EXTENSIONS = {
    "html", "htm", "php", "asp", "aspx", "jsp", "cgi", "pl", "exe",
    "zip", "rar", "js", "bat", "cmd", "scr", "msi", "apk", "dll",
}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc",
}

_IP_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
_PUNYCODE_RE = re.compile(r"xn--", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _entropy(s: str) -> float:
    """Shannon entropy (bits per character) of a string."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _ratio(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


# ---------------------------------------------------------------------------
# Lexical features
# ---------------------------------------------------------------------------
def extract_lexical(url: str) -> dict:
    u = url.strip()
    low = u.lower()
    parsed = urlparse(u)
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    n = len(u)
    num_digits = sum(c.isdigit() for c in u)
    num_special = sum(not c.isalnum() and not c.isspace() for c in u)

    # File extension of the last path segment.
    last_seg = path.rstrip("/").split("/")[-1] if path else ""
    ext = last_seg.rsplit(".", 1)[-1].lower() if "." in last_seg else ""
    has_file_ext = int(ext in FILE_EXTENSIONS)

    # Query parameters: `?a=1&b=2` -> 2 params (empty query -> 0).
    num_query_params = len([p for p in query.split("&") if p]) if query else 0

    # Redirect / protocol-relative patterns and explicit ports.
    has_redirect = int("//" in u[8:] if low.startswith(("http://", "https://")) else "//" in u)
    has_port = int(parsed.port is not None)

    return {
        # --- original lexical features ---
        "url_length": n,
        "num_dots": u.count("."),
        "num_slashes": u.count("/"),
        "num_digits": num_digits,
        "num_hyphens": u.count("-"),
        "num_underscores": u.count("_"),
        "num_at": u.count("@"),
        "num_question": u.count("?"),
        "num_equals": u.count("="),
        "num_ampersand": u.count("&"),
        "has_https": int(low.startswith("https")),
        "has_ip": int(bool(_IP_RE.search(u))),
        "suspicious_word_count": sum(w in low for w in SUSPICIOUS_WORDS),
        "path_length": len(path),
        "query_length": len(query),
        # --- new lexical features ---
        "url_entropy": round(_entropy(u), 4),
        "digit_ratio": round(_ratio(num_digits, n), 4),
        "special_char_ratio": round(_ratio(num_special, n), 4),
        "path_depth": len([p for p in path.split("/") if p]),
        "has_file_ext": has_file_ext,
        "num_query_params": num_query_params,
        "fragment_length": len(fragment),
        "has_redirect": has_redirect,
        "has_port": has_port,
        "is_punycode": int(bool(_PUNYCODE_RE.search(u))),
    }


# ---------------------------------------------------------------------------
# Host features
# ---------------------------------------------------------------------------
def extract_host(url: str) -> dict:
    ext = tldextract.extract(url)
    domain = ext.domain or ""
    subdomain = ext.subdomain or ""
    suffix = (ext.suffix or "").lower()
    registered = ext.registered_domain or ""

    low = url.lower()
    brand_in_subdomain = int(
        any(b in subdomain.lower() for b in BRANDS)
    )
    brand_in_path = int(any(b in low.split(registered.lower(), 1)[-1] for b in BRANDS)) if registered else 0

    return {
        # --- original host features ---
        "domain_length": len(domain),
        "subdomain_count": subdomain.count(".") + (1 if subdomain else 0),
        "has_subdomain": int(bool(subdomain)),
        "tld_length": len(suffix),
        "registered_domain_length": len(registered),
        "hyphens_in_domain": domain.count("-"),
        "digits_in_domain": sum(c.isdigit() for c in domain),
        "domain_entropy": round(_entropy(domain), 4),
        # --- new host features ---
        "suspicious_tld": int(suffix in SUSPICIOUS_TLDS),
        "brand_in_subdomain": brand_in_subdomain,
        "brand_in_path": brand_in_path,
        "hyphens_in_subdomain": subdomain.count("-"),
        "domain_digit_ratio": round(_ratio(sum(c.isdigit() for c in domain), len(domain)), 4),
        "is_url_shortener": int(registered.lower() in URL_SHORTENERS),
        "subdomain_entropy": round(_entropy(subdomain), 4),
    }


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------
def extract_all(url: str) -> dict:
    feats: dict = {}
    feats.update(extract_lexical(url))
    feats.update(extract_host(url))
    return feats


LEXICAL_COLS = list(extract_lexical("http://example.com").keys())
HOST_COLS = list(extract_host("http://example.com").keys())
ALL_COLS = LEXICAL_COLS + HOST_COLS

if __name__ == "__main__":
    import json, sys

    target = sys.argv[1] if len(sys.argv) > 1 else "http://example.com/login"
    print(json.dumps(extract_all(target), indent=2))
    print(f"\n{len(LEXICAL_COLS)} lexical + {len(HOST_COLS)} host = {len(ALL_COLS)} features")
