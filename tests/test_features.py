"""Tests for feature extraction and input validation."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features import (  # noqa: E402
    ALL_COLS,
    HOST_COLS,
    LEXICAL_COLS,
    explain_reasons,
    extract_all,
    extract_host,
    extract_lexical,
    normalize_url,
)


class TestFeatureShape(unittest.TestCase):
    def test_column_groups_add_up(self):
        self.assertEqual(len(LEXICAL_COLS), 25)
        self.assertEqual(len(HOST_COLS), 15)
        self.assertEqual(len(ALL_COLS), 40)
        self.assertEqual(len(set(ALL_COLS)), 40, "feature names must be unique")

    def test_extract_all_returns_all_columns(self):
        feats = extract_all("https://www.example.com/login")
        self.assertEqual(list(feats.keys()), ALL_COLS)

    def test_value_types(self):
        feats = extract_all("https://www.example.com/login")
        for key, value in feats.items():
            self.assertIsInstance(value, (int, float), f"{key} has type {type(value)}")


class TestLexicalCorrectness(unittest.TestCase):
    def test_https_requires_scheme(self):
        self.assertEqual(extract_lexical("https://example.com")["has_https"], 1)
        self.assertEqual(extract_lexical("http://example.com")["has_https"], 0)
        # the old bug: a hostname merely starting with "https"
        self.assertEqual(extract_lexical("httpswhatever.example.com")["has_https"], 0)

    def test_ip_only_in_hostname(self):
        self.assertEqual(extract_lexical("http://192.168.1.10/login")["has_ip"], 1)
        self.assertEqual(extract_lexical("http://[::1]/login")["has_ip"], 1)
        # IP-like text in the query must NOT count
        self.assertEqual(extract_lexical("http://example.com/?ip=999.999.999.999")["has_ip"], 0)
        # invalid IP literal
        self.assertEqual(extract_lexical("http://999.999.999.999/login")["has_ip"], 0)

    def test_malformed_port_does_not_crash(self):
        for url in ("https://example.com:abc/path", "https://example.com:99999/path"):
            feats = extract_all(url)  # must not raise
            self.assertEqual(feats["has_port"], 0)

    def test_valid_port_detected(self):
        self.assertEqual(extract_lexical("https://example.com:8443/login")["has_port"], 1)

    def test_suspicious_words_are_distinct(self):
        url = "http://login-login-login-verify.example.com"
        self.assertEqual(extract_lexical(url)["distinct_suspicious_words"], 2)

    def test_file_extension(self):
        self.assertEqual(extract_lexical("http://example.com/a/b.php")["has_file_ext"], 1)
        self.assertEqual(extract_lexical("http://example.com/a/b.html")["has_file_ext"], 1)
        self.assertEqual(extract_lexical("http://example.com/a/b")["has_file_ext"], 0)

    def test_path_and_query(self):
        feats = extract_lexical("http://example.com/a/b/c?x=1&y=2#frag")
        self.assertEqual(feats["path_depth"], 3)
        self.assertEqual(feats["num_query_params"], 2)
        self.assertEqual(feats["fragment_length"], 4)


class TestHostFeatures(unittest.TestCase):
    def test_suspicious_tld(self):
        self.assertEqual(extract_host("http://evil.example.click/login")["suspicious_tld"], 1)
        self.assertEqual(extract_host("https://example.com")["suspicious_tld"], 0)

    def test_brand_in_subdomain(self):
        self.assertEqual(
            extract_host("http://paypal.secure-login.xyz/")["brand_in_subdomain"], 1
        )
        self.assertEqual(
            extract_host("https://www.paypal.com/")["brand_in_subdomain"], 0
        )

    def test_subdomain_count(self):
        self.assertEqual(extract_host("https://a.b.example.com")["subdomain_count"], 2)
        self.assertEqual(extract_host("https://example.com")["subdomain_count"], 0)


class TestExplainReasons(unittest.TestCase):
    def test_phishing_reasons_are_plain_english(self):
        reasons = explain_reasons(
            "http://paypal-secure-login.example.com/verify/account", True
        )
        self.assertTrue(reasons)
        self.assertLessEqual(len(reasons), 4)
        joined = " ".join(reasons)
        self.assertIn("HTTP", joined)
        self.assertIn("brand", joined)
        for name in ALL_COLS:
            self.assertNotIn(name, joined, f"raw feature name leaked: {name}")

    def test_safe_reasons_mention_https(self):
        reasons = explain_reasons("https://www.google.com", False)
        self.assertTrue(any("HTTPS" in r for r in reasons))
        self.assertLessEqual(len(reasons), 4)

    def test_reason_limit(self):
        self.assertEqual(
            len(explain_reasons("http://192.168.1.10/login.php", True, max_reasons=2)),
            2,
        )

    def test_always_returns_something(self):
        # Forced-phishing verdict on an otherwise clean URL must still explain.
        self.assertTrue(explain_reasons("https://www.google.com", True))
        self.assertTrue(explain_reasons("https://www.google.com", False))


class TestNormalizeUrl(unittest.TestCase):
    def test_adds_scheme(self):
        self.assertEqual(normalize_url("example.com/login"), "http://example.com/login")

    def test_rejects_empty(self):
        for bad in ("", "   ", None, 123):
            with self.assertRaises(ValueError):
                normalize_url(bad)

    def test_rejects_malformed_port(self):
        with self.assertRaises(ValueError):
            normalize_url("https://example.com:abc/path")

    def test_rejects_missing_host(self):
        with self.assertRaises(ValueError):
            normalize_url("http:///path")

    def test_rejects_whitespace(self):
        for bad in ("not a url", "https://exa mple.com", "https://example.com/a b"):
            with self.assertRaises(ValueError):
                normalize_url(bad)

    def test_accepts_valid(self):
        self.assertEqual(
            normalize_url("https://example.com:8443/a"), "https://example.com:8443/a"
        )


if __name__ == "__main__":
    unittest.main()
