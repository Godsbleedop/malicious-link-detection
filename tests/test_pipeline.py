"""End-to-end tests for the shared prediction pipeline.

These tests need a trained ``model.pkl``. If it is missing they are skipped, so
the feature tests still run on a fresh checkout.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier import DEFAULT_MODEL_PATH, classify, load_bundle  # noqa: E402


@unittest.skipUnless(DEFAULT_MODEL_PATH.exists(), "model.pkl not trained yet")
class TestClassifierPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle(DEFAULT_MODEL_PATH)

    def test_returns_expected_keys(self):
        result = classify("https://www.google.com", self.bundle)
        for key in ("url", "label", "is_phishing", "p_phishing",
                    "confidence", "threshold", "features", "reasons", "model_name"):
            self.assertIn(key, result)
        self.assertIsInstance(result["is_phishing"], bool)
        self.assertGreaterEqual(result["p_phishing"], 0.0)
        self.assertLessEqual(result["p_phishing"], 1.0)

    def test_feature_order_matches_bundle(self):
        result = classify("https://www.google.com", self.bundle)
        self.assertEqual(list(result["features"].columns), list(self.bundle["features"]))

    def test_known_legitimate_url(self):
        for url in ("https://www.google.com", "https://stackoverflow.com/questions/1"):
            self.assertFalse(classify(url, self.bundle)["is_phishing"], url)

    def test_known_phishing_urls(self):
        for url in (
            "http://paypal-secure-login.example.com/verify/account",
            "http://192.168.1.10/login.php",
            "https://secure-bank-update.free-bonus.click/verify?user=abc",
        ):
            self.assertTrue(classify(url, self.bundle)["is_phishing"], url)

    def test_threshold_is_respected(self):
        url = "https://www.google.com"
        result = classify(url, self.bundle)
        p = result["p_phishing"]
        # force an impossible threshold -> must be flagged
        self.assertTrue(classify(url, self.bundle, threshold=0.0)["is_phishing"])
        self.assertFalse(classify(url, self.bundle, threshold=1.0)["is_phishing"])
        # default behaviour must agree with the saved threshold
        self.assertEqual(result["is_phishing"], p >= result["threshold"])

    def test_invalid_url_raises(self):
        with self.assertRaises(ValueError):
            classify("https://example.com:abc/path", self.bundle)
        with self.assertRaises(ValueError):
            classify("", self.bundle)

    def test_schemeless_url_is_normalised(self):
        result = classify("google.com", self.bundle)
        self.assertTrue(result["url"].startswith("http://"))


if __name__ == "__main__":
    unittest.main()
