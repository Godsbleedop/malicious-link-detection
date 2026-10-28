# 🔗 Malicious URL Classification

Classify a URL as **phishing (malicious)** or **legitimate (safe)** using
hand-crafted lexical + host features and classical machine learning.

This document is the single source of truth for the project: setup, features,
methodology, results, usage, limitations and a dated changelog.

---

## 1. Overview

Phishing links are the entry point for most credential-theft attacks. This
project builds a lightweight, explainable classifier that looks at the URL
string itself (no page loading required) and flags suspicious links before a
user clicks.

The project is a postgraduate mini-project comparing four classical models
(Logistic Regression, SVM, Random Forest, XGBoost) across three feature
configurations (lexical-only, host-only, combined).

**Headline result (official test split):**

| Metric | Value |
|---|---|
| Accuracy | **95.06 %** |
| Precision | 93.63 % |
| Recall | **95.32 %** |
| F1 | 94.47 % |
| ROC-AUC | **0.9914** |

Best model: **XGBoost** with a validation-tuned decision threshold of **0.45**.

---

## 2. Dataset

Source: Phishing Attack Dataset (Phishtank phishing + CommonCrawl legitimate),
provided in `dephides/dataset/small_dataset/`.

| Split | Phishing | Legitimate | Total |
|---|---|---|---|
| train.txt | 162,463 | 201,736 | 364,199 |
| val.txt | 46,649 | 57,927 | 104,576 |
| test.txt | 22,978 | 28,532 | 51,510 |
| **Overall** | **232,090** | **288,195** | **520,285** |

- Format: tab-separated `label<TAB>url` (`phishing` / `legitimate`).
- Class balance ≈ 44.6 % phishing — mildly imbalanced (no resampling needed;
  `class_weight="balanced"` is used where supported).
- The big (5.2 M row) dataset is not available locally.

### Sampling used for training
| Split | Rows used |
|---|---|
| train | 50,000 |
| val | 15,000 |
| test | 20,000 |

Sampling keeps training laptop-friendly; all splits remain stratified.

---

## 3. Feature Engineering (40 features)

Features are extracted from the URL string only — no network requests, no page
content. Implemented in `features.py`.

### 3.1 Lexical features (25)
Properties of the raw URL string.

| # | Feature | Description |
|---|---|---|
| 1 | `url_length` | total characters |
| 2–10 | `num_dots`, `num_slashes`, `num_digits`, `num_hyphens`, `num_underscores`, `num_at`, `num_question`, `num_equals`, `num_ampersand` | character counts |
| 11 | `has_https` | scheme is HTTPS |
| 12 | `has_ip` | IPv4 address in the URL |
| 13 | `distinct_suspicious_words` | number of **distinct** configured keywords present (login, verify, bank, paypal, …; not total occurrences) |
| 14–15 | `path_length`, `query_length` | lengths of path / query |
| 16 | `url_entropy` | Shannon entropy of the whole URL (randomness) |
| 17 | `digit_ratio` | digits / length |
| 18 | `special_char_ratio` | non-alphanumeric / length |
| 19 | `path_depth` | number of path segments |
| 20 | `has_file_ext` | extension is in a common web/executable list (`.php`, `.js`, `.exe`, …) — weak signal only, not proof of malice |
| 21 | `num_query_params` | number of `&`-separated parameters |
| 22 | `fragment_length` | characters after `#` |
| 23 | `has_redirect` | protocol-relative / `//` pattern |
| 24 | `has_port` | explicit port number |
| 25 | `is_punycode` | contains `xn--` (homograph/IDN attack) |

### 3.2 Host features (15)
Properties of the host / registered domain (via `tldextract`).

| # | Feature | Description |
|---|---|---|
| 26–33 | `domain_length`, `subdomain_count`, `has_subdomain`, `tld_length`, `registered_domain_length`, `hyphens_in_domain`, `digits_in_domain`, `domain_entropy` | original host features |
| 34 | `suspicious_tld` | TLD in abuse list (`.tk`, `.ml`, `.click`, `.xyz`, `.top`, …) |
| 35 | `brand_in_subdomain` | impersonated brand in subdomain (`paypal.secure.xyz`) |
| 36 | `brand_in_path` | brand name in path |
| 37 | `hyphens_in_subdomain` | hyphen count in subdomain |
| 38 | `domain_digit_ratio` | digits / domain length |
| 39 | `is_url_shortener` | known shortener (bit.ly, tinyurl, …) |
| 40 | `subdomain_entropy` | Shannon entropy of subdomain |

> **Why these features matter:** the earlier version used only 23 simple counts
> and missed obvious phishing such as `secure-bank-update.free-bonus.click`.
> The new entropy, suspicious-TLD and brand-impersonation features catch exactly
> these patterns.

`extract_all(url)` merges both groups; `ALL_COLS` is the canonical column order
shared by training, evaluation and prediction.

---

## 4. Methodology

```
load split → sample → extract 40 features → standardise (LR/SVM)
          → train 4 models → 3-fold CV → select by validation F1
          → tune threshold on validation → evaluate on official test split
          → save bundle (model, features, scaler, threshold)
```

### 4.1 Models
| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=2000`, `class_weight="balanced"` |
| SVM (RBF) | `kernel="rbf"`, `probability=True`, `class_weight="balanced"` (trained on 12k subsample for speed) |
| Random Forest | `n_estimators=300`, `class_weight="balanced_subsample"` |
| XGBoost | `n_estimators=300`, `max_depth=6`, `lr=0.1`, `subsample=0.9`, `colsample_bytree=0.9` |

LR and SVM consume standardised features; tree models use raw features.

### 4.2 Model selection & threshold
- Models are compared on the **validation** split (never the test split).
- The best model by validation F1 is chosen (**XGBoost**).
- The decision threshold is tuned on validation probabilities in `[0.05, 0.95]`:
  - **F1-optimal threshold** → `0.45`
  - **High-recall threshold** (recall ≥ 0.95) → `0.44`
- The threshold is stored in `model.pkl` and applied automatically by
  `predict.py` and `app.py`. `--high-recall` selects the higher-recall point.

> This directly addresses the earlier weakness: predictions no longer use a
> hard-coded 0.5 cutoff.

---

## 5. Results

### 5.1 Model comparison (validation, threshold 0.5)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9239 | 0.9059 | 0.9264 | 0.9160 | 0.9815 |
| SVM (RBF) | 0.9396 | 0.9271 | 0.9392 | 0.9331 | 0.9864 |
| Random Forest | 0.9437 | 0.9388 | 0.9355 | 0.9371 | 0.9892 |
| **XGBoost** | **0.9492** | **0.9463** | 0.9401 | **0.9432** | **0.9917** |

### 5.2 3-fold cross-validation (subset)
| Model | F1 mean ± std | ROC-AUC mean ± std |
|---|---|---|
| Logistic Regression | 0.9097 ± 0.0043 | 0.9790 ± 0.0011 |
| Random Forest | 0.9277 ± 0.0029 | 0.9855 ± 0.0012 |
| XGBoost | **0.9321 ± 0.0016** | **0.9878 ± 0.0004** |

SVM CV was skipped (too slow); it was trained on a 12k subsample.

### 5.3 Official test split — best model (XGBoost)
| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | False-negative rate |
|---|---|---|---|---|---|---|
| 0.50 (default) | 0.9514 | 0.9444 | 0.9459 | 0.9451 | 0.9914 | 5.41 % |
| **0.45 (tuned)** | 0.9506 | 0.9363 | **0.9532** | **0.9447** | **0.9914** | **4.68 %** |
| 0.44 (high-recall) | 0.9503 | 0.9345 | 0.9546 | 0.9445 | 0.9914 | 4.54 % |

**Interpretation:** the tuned threshold trades ~1 point of precision for
~0.7 points of recall, cutting missed phishing from 5.41 % to 4.68 %. For
security, that is usually the right trade-off.

### 5.4 Feature configuration comparison (XGBoost, test split)
| Config | Features | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Lexical-only | 25 | 0.9234 | 0.9072 | 0.9213 | 0.9142 | 0.9804 |
| Host-only | 15 | 0.8160 | 0.8557 | 0.7028 | 0.7718 | 0.8838 |
| **Combined** | **40** | **0.9466** | **0.9365** | **0.9432** | **0.9398** | **0.9899** |

Combined features clearly win: lexical features carry most of the signal,
host features add complementary domain-level evidence.

### 5.5 Figures
| Figure | File |
|---|---|
| ROC curve | `docs/figures/roc_curve.png` |
| Precision–Recall curve | `docs/figures/pr_curve.png` |
| Confusion matrix | `docs/figures/confusion_matrix.png` |
| Feature importances | `docs/figures/feature_importance.png` |
| Config comparison | `docs/figures/model_comparison.png` |

---

## 6. Usage

### Setup
```bash
cd /home/aaron/MTECH/aismp
pip install -r requirements.txt
```

### Train (all models, CV, threshold tuning, save bundle)
```bash
python3 train_model.py                 # defaults: 50k train / 15k val / 20k test
python3 train_model.py --train-size 0  # use the full training split
python3 train_model.py --no-cv         # skip cross-validation (faster)
```

### Evaluate & generate figures
```bash
python3 evaluate.py     # writes metrics_configs.json + docs/figures/*
```

### Predict (CLI)
```bash
python3 predict.py "https://example.com"
python3 predict.py --loop                       # interactive loop
python3 predict.py --high-recall "http://..."   # favour recall
```

### Web UI

**Option A — zero-dependency (works immediately):**
```bash
python3 web_app.py       # open http://localhost:8000
```
Single-page UI with a URL input, the verdict, a confidence percentage and a few
plain-English reasons. Uses only the Python standard library and exposes no
model or feature details.

**Option B — Streamlit (optional):**
```bash
pip install streamlit     # install in a venv if the system Python is managed
streamlit run app.py
```
Same interface: input, verdict, confidence and plain-English reasons.

### Plain-English explanations
Every verdict comes with up to four short reasons, shown in the CLI and both
UIs. They describe visible warning signs rather than model internals, e.g.:

- `http://paypal-secure-login.example.com/verify/account` →
  *"Uses an insecure HTTP connection instead of HTTPS"*, *"Puts a well-known
  brand name in the subdomain"*, *"Contains words commonly used to trick people
  (login, verify, secure)"*.
- `https://www.google.com` →
  *"Uses a secure HTTPS connection"*, *"Uses a normal domain name rather than an
  IP address"*.

The reasons are heuristic indicators, not a SHAP-style attribution of the
model's exact decision.

### Quick feature dump
```bash
python3 features.py "http://paypal-secure-login.free-bonus.click/verify"
```

### Run the automated tests
```bash
python3 -m unittest discover -s tests -v
```
30 tests cover feature correctness (scheme detection, IP validation, malformed
ports, feature order/types, explanation text) and the end-to-end prediction
pipeline (known legitimate/phishing URLs, threshold behaviour, invalid-input
handling).

### Error handling
- `features.normalize_url()` validates user input at the boundary: empty input,
  whitespace/control characters, missing hostnames and malformed ports
  (e.g. `https://example.com:abc`) raise a clear `ValueError`.
- Feature extraction itself never crashes — malformed ports are mapped to
  `has_port = 0`, and invalid IP literals to `has_ip = 0`.
- The CLI prints `Invalid URL: …` and continues; the web UI renders an error box.
- `predict.py` exits with a helpful message if `model.pkl` is missing.

---

## 7. Project Files

| Path | Purpose |
|---|---|
| `features.py` | 40 lexical + host features, input validation, explanation helper, `ALL_COLS` |
| `classifier.py` | shared load/classify logic (CLI + both UIs) |
| `train_model.py` | train/compare models, CV, threshold tuning, save `model.pkl` |
| `predict.py` | CLI prediction with tuned threshold, reasons + loop mode |
| `app.py` | Streamlit web UI (optional) |
| `web_app.py` | zero-dependency stdlib web UI (default demo) |
| `evaluate.py` | test-set evaluation, config comparison, figures |
| `tests/` | 30 automated tests (`unittest`) |
| `model.pkl` | bundle: model, features, scaler, threshold, metadata |
| `metrics.json` | validation/test metrics from training |
| `metrics_configs.json` | feature-config comparison results |
| `docs/figures/` | ROC, PR, confusion matrix, importance, comparison plots |
| `docs/PROJECT_ORIGINAL.md` | original project notes (superseded by README) |
| `docs/261091010001-22_miniproject-3.pdf` | assignment brief |
| `notebooks/analysis_executed.ipynb` | EDA / feature-selection notebook (earlier work) |
| `notebooks/full_pipeline_executed.ipynb` | 4-model comparison notebook (earlier work) |
| `README.md` | this document |
| `PROGRESS.md` | dated changelog |
| `dephides/` | reference repo + dataset (not part of the deliverable) |

---

## 8. Limitations & Future Work

**Limitations**
- Features are URL-only; no page content, TLS certificate, WHOIS age, or
  redirect-chain analysis.
- Dataset is ~520k rows from a fixed snapshot; real phishing evolves quickly.
- SVM does not scale to the full dataset in reasonable time.
- No adversarial/obfuscated URL hardening (e.g., homoglyph look-alikes beyond
  basic punycode detection).
- `tldextract` may need a network fetch on first run (public suffix list).

**Future work**
- Add domain-age / WHOIS and certificate features.
- Add SHAP per-URL explanations (not just global importance).
- Calibrate probabilities (`CalibratedClassifierCV`) for better confidence.
- Scale to the 5.2M-row dataset; add deep-learning baselines (CNN/URLNet).
- Deploy as a REST API and browser extension.
- Continual retraining on fresh phishing feeds.

---

## 9. Changelog

### 2026-10-26 — Code review fixes & cleanup
- **`has_https` fixed:** now checks `parsed.scheme == "https"` instead of
  `str.startswith("https")` (so `httpswhatever.example.com` is no longer HTTPS).
- **IP detection fixed:** validates the URL *hostname* with `ipaddress`; IP-like
  text in paths/queries and invalid literals (`999.999.999.999`) no longer count.
- **Malformed ports no longer crash:** `has_port` uses a safe accessor, and the
  new `normalize_url()` validates input at the boundary with a clear error.
- **Feature rename:** `suspicious_word_count` → `distinct_suspicious_words`
  (it counts distinct keywords, not occurrences).
- **Docs:** clarified that `has_file_ext` is a weak signal, not proof of malice.
- **Reusability:** new `classifier.py` is the single prediction path used by the
  CLI, the stdlib web UI and the Streamlit app (no duplicated logic).
- **Testing:** added `tests/` with **25 `unittest` tests** (features + pipeline).
- **Cleanup:** removed duplicate notebooks, notebook generators, old training log
  and caches; moved notebooks to `notebooks/`, docs to `docs/`.
- **Retrained:** F1-optimal threshold now **0.45**, high-recall **0.44**; test
  accuracy 95.06 %, recall 95.32 %, ROC-AUC 0.9914.

### 2026-10-24 — Major upgrade
- **Features:** 23 → **40** (added entropy, ratios, path depth, file extension,
  query params, redirect, port, punycode, suspicious TLD, brand impersonation,
  URL shortener, subdomain entropy, etc.).
- **Training:** `train_model.py` rewritten to train all 4 models, run 3-fold CV,
  select by validation F1, and tune the decision threshold.
- **Threshold:** F1-optimal (`0.39`) and high-recall (`0.43`) thresholds saved in
  `model.pkl`; `predict.py` applies them instead of 0.5.
- **Evaluation:** `evaluate.py` added — official test split, ROC/PR curves,
  confusion matrix, feature importance, lexical/host/combined comparison.
- **UI:** zero-dependency web UI (`web_app.py`) and Streamlit app (`app.py`).
- **Prediction:** `predict.py` now supports `--loop` and `--high-recall`.
- **Docs:** this README created; `metrics.json` / `metrics_configs.json` added.
- **Result:** fixed the previously-missed `secure-bank-update.free-bonus.click`
  URL (now flagged as phishing at 99.90 %).

### Earlier work (pre-2026-10-24)
- Dataset validation and cleaning.
- Initial 23 features and Random Forest baseline.
- Notebook exploration and comparison of LR / SVM / RF / XGBoost.
