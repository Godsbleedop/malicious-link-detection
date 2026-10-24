# Progress Report — Malicious URL Classification

> Master documentation lives in [`README.md`](README.md). This file is the
> dated changelog.

## 2026-10-24 — Major upgrade

### Feature engineering (23 → 40 features)
- Lexical (25): added `url_entropy`, `digit_ratio`, `special_char_ratio`,
  `path_depth`, `has_file_ext`, `num_query_params`, `fragment_length`,
  `has_redirect`, `has_port`, `is_punycode`.
- Host (15): added `suspicious_tld`, `brand_in_subdomain`, `brand_in_path`,
  `hyphens_in_subdomain`, `domain_digit_ratio`, `is_url_shortener`,
  `subdomain_entropy`.
- Expanded suspicious keywords (28) and suspicious TLD list.

### Training (`train_model.py`)
- Trains LR, SVM (RBF), Random Forest and XGBoost.
- 3-fold cross-validation on a 12k subset (SVM skipped — too slow).
- Model selection by **validation F1** (no test leakage).
- Threshold tuning on validation: F1-optimal **0.39**, high-recall **0.43**.
- Saves one bundle: `{model, features, scaler, threshold, metadata}`.

### Results (official test split, XGBoost)
| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | FNR |
|---|---|---|---|---|---|---|
| 0.50 | 0.9518 | 0.9458 | 0.9453 | 0.9455 | 0.9915 | 5.47 % |
| 0.39 (tuned) | 0.9487 | 0.9264 | 0.9606 | 0.9432 | 0.9915 | **3.94 %** |
| 0.43 (high-recall) | 0.9503 | 0.9337 | 0.9556 | 0.9445 | 0.9915 | 4.44 % |

Feature configs (test): lexical-only F1 0.9150 / AUC 0.9803,
host-only F1 0.7718 / AUC 0.8838, **combined F1 0.9402 / AUC 0.9900**.

### New files
- `evaluate.py` — test evaluation, config comparison, figures.
- `app.py` — Streamlit web UI with threshold slider.
- `web_app.py` — zero-dependency stdlib web UI (runs without extra installs).
- `metrics.json`, `metrics_configs.json` — machine-readable results.
- `docs/figures/*.png` — ROC, PR, confusion matrix, importance, comparison.
- `README.md` — consolidated documentation.
- `requirements.txt` — dependency list.

### Fixes
- The previously missed `https://secure-bank-update.free-bonus.click/verify?user=abc`
  is now correctly classified as **PHISHING (99.90 %)**.
- `predict.py` applies the tuned threshold (not 0.5) and supports `--loop` /
  `--high-recall`.

---

## Earlier work (pre-2026-10-24)

### Dataset validation
- Loaded `dephides/dataset/small_dataset/{train,val,test}.txt`.
- Verified counts match the brief: 520,285 total (232,090 phishing /
  288,195 legitimate), 0.7/0.2/0.1 split.
- Cleaned malformed lines; class balance ≈ 44.6 % phishing.

### Initial features (23)
- 15 lexical (lengths, character counts, HTTPS, IP, keywords, path/query).
- 8 host (domain length, subdomains, TLD, hyphens/digits, entropy).

### Initial models
- Baseline Logistic Regression + Random Forest on combined features.
- Random Forest strongest baseline; later compared with SVM and XGBoost in
  `full_pipeline.ipynb`.

### Next steps (superseded by the 2026-10-24 upgrade)
- ~~Add SVM/XGBoost~~ ✅
- ~~Compare feature configs~~ ✅
- ~~Threshold tuning~~ ✅
- Remaining: report/slides, WHOIS/domain-age features, SHAP per-URL explanations.
