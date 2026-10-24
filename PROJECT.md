# Malicious URL Classification — Project Documentation

## 1. What this project does
Given a URL, this project predicts whether it is **phishing (malicious)** or **legitimate (safe)**
using classical machine-learning on hand-crafted **lexical** and **host-based** features.
This matches the mini-project brief (PDF): compare Logistic Regression, SVM, Random Forest,
XGBoost across lexical-only / host-only / combined feature configs.

## 2. Quick start
```bash
cd /home/aaron/MTECH/aismp
pip install -r requirements.txt          # pandas, scikit-learn, tldextract, joblib, matplotlib
python3 train_model.py                   # trains & saves model.pkl (~1 min on 50k samples)
python3 predict.py "https://example.com"  # interactive: python3 predict.py
```

## 3. Project files
| File | Purpose |
|---|---|
| `features.py` | Feature extraction (lexical + host) — reusable module |
| `train_model.py` | Loads dataset, trains RandomForest, saves `model.pkl` |
| `predict.py` | CLI: takes a URL, prints prediction + confidence |
| `model.pkl` | Trained model + feature list (joblib) |
| `analysis.ipynb` / `analysis_executed.ipynb` | EDA, feature selection, baseline comparison |
| `PROGRESS.md` | Weekly progress report |
| `PROJECT.md` | This file |

## 4. How the code works
### features.py
- `extract_lexical(url)` → 15 features: length, counts of `. / digits - _ @ ? = &`,
  HTTPS flag, IP-in-host flag, suspicious-keyword count, path/query lengths.
- `extract_host(url)` → 8 features via `tldextract`: domain length, subdomain count,
  TLD length, hyphens/digits in domain, domain entropy.
- `extract_all(url)` merges both; `ALL_COLS` is the canonical column order used
  everywhere (training, prediction, notebook).

### train_model.py
1. Reads `dephides/dataset/small_dataset/train.txt` (tab-separated `label\turl`),
   skips blank/malformed lines.
2. Samples 50,000 rows (for laptop-friendly training).
3. Builds feature matrix with `extract_all`.
4. 80/20 stratified split, trains `RandomForestClassifier(n_estimators=200)`.
5. Saves `{'model': model, 'features': ALL_COLS}` to `model.pkl`.

### predict.py
1. Loads `model.pkl`.
2. Extracts features for the input URL in the same column order.
3. Prints `PHISHING (malicious)` or `LEGITIMATE (safe)` with confidence.

## 5. Dataset
- Source: Phishing Attack Dataset (Phishtank phishing + CommonCrawl legitimate).
- Small: 520,285 samples (232,090 phishing / 288,195 legitimate), split 0.7/0.2/0.1.
- Big: 5,202,841 samples (train.txt unavailable locally — Google Drive access needed).

## 6. What you need to learn (beginner roadmap)
1. **Python basics** — variables, functions, lists, dicts, file I/O, `if/for`.
   Free: Kaggle "Python" micro-course, freeCodeCamp YouTube.
2. **pandas** — DataFrames, `read_csv`, filtering, `groupby`, `.apply`.
   Free: Kaggle "Pandas" micro-course.
3. **NumPy** — arrays, basic math (used under the hood by sklearn).
4. **scikit-learn** — `train_test_split`, `fit`, `predict`, metrics,
   `RandomForestClassifier`, `LogisticRegression`, `SVC`, `XGBClassifier`.
5. **Feature engineering for URLs** — `urllib.parse`, `tldextract`, regex,
   entropy, keyword matching.
6. **Evaluation** — accuracy, precision, recall, F1, confusion matrix.
7. **Jupyter notebooks** — for exploration and presentation.
8. **joblib** — saving/loading trained models.

## 7. Next steps (next week)
- Add SVM and XGBoost models.
- Compare lexical-only / host-only / combined feature configs.
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV).
- Evaluate on the official test split; finalize report + slides.
- Optionally scale to the big dataset once `train.txt` is obtained.