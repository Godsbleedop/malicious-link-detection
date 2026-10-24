import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# Malicious URL Classification — Initial Progress
**Dataset:** Phishing Attack Dataset (small, 520,285 samples)
**Focus this week:** Dataset validation, feature engineering & selection, baseline models
**Next week:** SVM, XGBoost, full 3-config comparison, tuning, final report""")

code("""import pandas as pd, numpy as np, matplotlib.pyplot as plt
from features import extract_all, LEXICAL_COLS, HOST_COLS, ALL_COLS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import warnings; warnings.filterwarnings('ignore')
print('imports ok')""")

md("## 1. Load & validate dataset")
code("""rows = []
with open('dephides/dataset/small_dataset/train.txt') as f:
    for line in f:
        line = line.rstrip('\\n')
        if not line.strip():
            continue
        parts = line.split('\\t')
        if len(parts) != 2:
            parts = line.split()
        if len(parts) != 2:
            continue
        rows.append(parts)
df = pd.DataFrame(rows, columns=['label', 'url'])
print('shape:', df.shape)
print(df['label'].value_counts())
print('duplicates:', df['url'].duplicated().sum())
df.head()""")

code("""# label distribution plot
vc = df['label'].value_counts()
fig, ax = plt.subplots(figsize=(5,4))
vc.plot(kind='bar', ax=ax, color=['#e74c3c','#2ecc71'])
ax.set_title('Class distribution (train set)')
ax.set_ylabel('count')
for i,v in enumerate(vc.values):
    ax.text(i, v+1000, f'{v:,}', ha='center')
plt.tight_layout(); plt.show()""")

md("## 2. Feature engineering")
code("""# sample for speed during prototyping
sample = df.sample(n=20000, random_state=42).reset_index(drop=True)
feats = pd.DataFrame([extract_all(u) for u in sample['url']])
feats['label'] = sample['label'].values
print('feature matrix:', feats.shape)
feats.head()""")

code("""# feature means by class
print(feats.groupby('label')[ALL_COLS].mean().T.round(3))""")

md("## 3. Feature selection (Random Forest importance)")
code("""X = feats[ALL_COLS]; y = (feats['label']=='phishing').astype(int)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=ALL_COLS).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8,7))
imp.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Feature importance (Random Forest)')
plt.tight_layout(); plt.show()
print('Top 10 features:')
print(imp.sort_values(ascending=False).head(10))""")

md("## 4. Baseline models (combined features)")
code("""Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
results = []
for name, model in [('LogisticRegression', LogisticRegression(max_iter=1000)),
                    ('RandomForest', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))]:
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, average='binary')
    results.append((name, acc, p, r, f1))
    print(f'{name}: acc={acc:.4f} prec={p:.4f} rec={r:.4f} f1={f1:.4f}')
    cm = confusion_matrix(yte, pred)
    ConfusionMatrixDisplay(cm, display_labels=['legitimate','phishing']).plot(cmap='Blues')
    plt.title(name); plt.show()
pd.DataFrame(results, columns=['model','accuracy','precision','recall','f1'])""")

md("## 5. Next-week plan")
md("""- Add SVM and XGBoost models.
- Compare **lexical-only**, **host-only**, **combined** feature configs.
- Hyperparameter tuning (GridSearch/RandomizedSearch).
- Evaluate on the official test split; produce final report + slides.
- Optional: scale to big_dataset if access granted.""")

nb['cells'] = cells
with open('/home/aaron/MTECH/aismp/analysis.ipynb', 'w') as f:
    nbf.write(nb, f)
print('notebook written')