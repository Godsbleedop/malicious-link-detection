"""Lexical and host-based feature extraction for malicious URL classification.

Feature groups
--------------
LEXICAL_COLS : properties of the raw URL string.
HOST_COLS    : properties of the host / registered domain (via tldextract).

`extract_all(url)` merges both and `ALL_COLS` is the canonical column order used
by training, evaluation and prediction.

Robustness
----------
Feature extraction is deliberately total: any string input yields a complete
feature row (malformed ports / invalid IPv6 are handled without raising). Input
*validation* is a separate concern and lives in :func:`normalize_url`, which
callers should run first so users get a clear error instead of a silent guess.
"""
from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import ParseResult, urlparse

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

# Common web / executable extensions. NOTE: a match is only a weak signal --
# `.php` or `.js` are used by millions of legitimate sites; the model weighs
# this feature together with the others, it does not imply maliciousness.
FILE_EXTENSIONS = {
    "html", "htm", "php", "asp", "aspx", "jsp", "cgi", "pl", "exe",
    "zip", "rar", "js", "bat", "cmd", "scr", "msi", "apk", "dll",
}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc",
}

_PUNYCODE_RE = re.compile(r"xn--", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Parsing helpers (never raise)
# ---------------------------------------------------------------------------
def _safe_parse(url: str) -> ParseResult:
    """urlparse that tolerates invalid IPv6 brackets etc."""
    try:
        return urlparse(url)
    except ValueError:
        return urlparse("")


def _safe_port(parsed: ParseResult) -> int:
    """1 if an explicit port is present and valid, else 0."""
    try:
        return int(parsed.port is not None)
    except ValueError:
        return 0


def _host_is_ip(parsed: ParseResult) -> int:
    """1 if the URL's hostname is a valid IPv4/IPv6 literal, else 0."""
    host = parsed.hostname
    if not host:
        return 0
    try:
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        return 0


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
# Input validation (call this at the user-input boundary)
# ---------------------------------------------------------------------------
def normalize_url(url: str) -> str:
    """Validate and normalise a user-supplied URL.

    - rejects non-strings and empty input,
    - rejects whitespace / control characters,
    - prepends ``http://`` when no scheme is given (``example.com``),
    - rejects URLs without a hostname,
    - rejects malformed / out-of-range ports with a clear message.

    Returns the normalised URL string, or raises ``ValueError``.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    u = url.strip()
    if not u:
        raise ValueError("URL is empty")
    if any(c.isspace() or ord(c) < 32 for c in u):
        raise ValueError("URL must not contain spaces or control characters")
    if "://" not in u:
        u = "http://" + u

    parsed = _safe_parse(u)
    if not parsed.hostname:
        raise ValueError(f"URL has no valid hostname: {url!r}")
    try:
        parsed.port  # noqa: B018 - property access validates the port
    except ValueError as exc:
        raise ValueError(f"Invalid port in URL {url!r}: {exc}") from exc
    return u


# ---------------------------------------------------------------------------
# Lexical features
# ---------------------------------------------------------------------------
def extract_lexical(url: str) -> dict:
    u = str(url).strip()
    low = u.lower()
    parsed = _safe_parse(u)
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

    # Protocol-relative / redirect patterns.
    if low.startswith(("http://", "https://")):
        has_redirect = int("//" in u.split("://", 1)[1])
    else:
        has_redirect = int("//" in u)

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
        # scheme check: only an actual https:// scheme counts
        "has_https": int(parsed.scheme.lower() == "https"),
        # validated IP literal in the hostname (not IP-like text in path/query)
        "has_ip": _host_is_ip(parsed),
        # number of *distinct* configured suspicious keywords present
        "distinct_suspicious_words": sum(w in low for w in SUSPICIOUS_WORDS),
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
        "has_port": _safe_port(parsed),
        "is_punycode": int(bool(_PUNYCODE_RE.search(u))),
    }


# ---------------------------------------------------------------------------
# Host features
# ---------------------------------------------------------------------------
def extract_host(url: str) -> dict:
    ext = tldextract.extract(str(url))
    domain = ext.domain or ""
    subdomain = ext.subdomain or ""
    suffix = (ext.suffix or "").lower()
    registered = ext.registered_domain or ""

    low = str(url).lower()
    brand_in_subdomain = int(any(b in subdomain.lower() for b in BRANDS))
    if registered:
        tail = low.split(registered.lower(), 1)[-1]
        brand_in_path = int(any(b in tail for b in BRANDS))
    else:
        brand_in_path = 0

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


def explain_reasons(url: str, is_phishing: bool, max_reasons: int = 4) -> list[str]:
    """Return short, plain-English reasons for a prediction.

    These describe visible indicators in the link (scheme, keywords, domain
    shape, etc.); they are not a mathematical attribution of the model's
    decision. Pass an already-normalised URL.
    """
    f = extract_all(url)
    low = str(url).lower()
    suffix = tldextract.extract(str(url)).suffix.lower()
    reasons: list[tuple[int, str]] = []

    if is_phishing:
        if not f["has_https"]:
            reasons.append((5, "Uses an insecure HTTP connection instead of HTTPS."))
        if f["has_ip"]:
            reasons.append((5, "Uses a raw IP address instead of a normal domain name."))
        if f["is_punycode"]:
            reasons.append((5, "Uses encoded characters that can disguise the real website name."))
        if f["brand_in_subdomain"]:
            reasons.append((5, "Puts a well-known brand name in the subdomain, a common impersonation trick."))
        if f["num_at"]:
            reasons.append((5, "Contains an '@' symbol, which can hide the real destination."))
        if f["suspicious_tld"]:
            reasons.append((3, f"Uses an uncommon domain ending (.{suffix}) often seen in phishing."))
        if f["is_url_shortener"]:
            reasons.append((3, "Uses a link shortener that hides the real destination."))
        words = [w for w in SUSPICIOUS_WORDS if w in low]
        if len(words) >= 2:
            reasons.append((3, "Contains words commonly used to trick people "
                               f"({', '.join(words[:3])})."))
        if f["subdomain_count"] >= 2 or f["subdomain_entropy"] >= 3.5:
            reasons.append((2, "Has a complex or random-looking subdomain."))
        if f["hyphens_in_domain"] >= 2:
            reasons.append((2, "The domain name contains several hyphens."))
        if f["hyphens_in_subdomain"] >= 2:
            reasons.append((2, "The subdomain contains several hyphens."))
        if f["digits_in_domain"] >= 3 or f["domain_digit_ratio"] >= 0.2:
            reasons.append((2, "The domain name contains an unusual number of digits."))
        if f["has_port"]:
            reasons.append((2, "Uses a non-standard network port."))
        if f["url_length"] >= 100:
            reasons.append((1, "The link is unusually long."))
        if f["num_query_params"] >= 4:
            reasons.append((1, "The link contains a long list of parameters."))
        if f["has_file_ext"]:
            reasons.append((1, "The link points to a downloadable or script file."))
    else:
        if f["has_https"]:
            reasons.append((5, "Uses a secure HTTPS connection."))
        if not f["has_ip"]:
            reasons.append((4, "Uses a normal domain name rather than an IP address."))
        if not f["is_punycode"]:
            reasons.append((4, "The domain name uses standard characters."))
        if not f["is_url_shortener"]:
            reasons.append((3, "Does not hide the destination behind a link shortener."))
        if not f["num_at"]:
            reasons.append((3, "Does not contain an '@' symbol that could hide the real destination."))
        if not f["suspicious_tld"]:
            reasons.append((2, "Uses a common domain ending."))
        if f["distinct_suspicious_words"] == 0:
            reasons.append((2, "Does not contain common phishing keywords."))
        if f["subdomain_count"] <= 1 and f["subdomain_entropy"] < 3.0:
            reasons.append((1, "The domain name is short and easy to read."))

    reasons.sort(key=lambda item: -item[0])
    out: list[str] = []
    for _, text in reasons:
        if text not in out:
            out.append(text)
        if len(out) >= max_reasons:
            break
    if not out:
        out.append(
            "The link shows patterns often associated with phishing."
            if is_phishing
            else "No strong warning signs were found in the link."
        )
    return out


LEXICAL_COLS = list(extract_lexical("http://example.com").keys())
HOST_COLS = list(extract_host("http://example.com").keys())
ALL_COLS = LEXICAL_COLS + HOST_COLS

if __name__ == "__main__":
    import json
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "http://example.com/login"
    print(json.dumps(extract_all(target), indent=2))
    print(f"\n{len(LEXICAL_COLS)} lexical + {len(HOST_COLS)} host = {len(ALL_COLS)} features")
