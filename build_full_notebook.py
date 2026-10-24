import nbformat as nbf
nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("# Malicious URL Classification — Full Pipeline\nRun all cells top-to-bottom. Trains LR, SVM, RF, XGBoost and reports Accuracy/Precision/Recall/F1 + ROC curves.")

code("""import pandas as pd, numpy as np, matplotlib.pyplot as plt, joblib
from features import extract_all, ALL_COLS, LEXICAL_COLS, HOST_COLS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, classification_report, ConfusionMatrixDisplay, confusion_matrix)
import warnings; warnings.filterwarnings('ignore')
print('imports ok')""")

md("## 1. Load & clean dataset")
code("""rows=[]
with open('dephides/dataset/small_dataset/train.txt') as f:
    for line in f:
        line=line.rstrip('\\n')
        if not line.strip(): continue
        parts=line.split('\\t')
        if len(parts)!=2: parts=line.split()
        if len(parts)!=2: continue
        rows.append(parts)
df=pd.DataFrame(rows, columns=['label','url'])
print('shape', df.shape)
print(df['label'].value_counts())""")

md("## 2. Feature engineering")
code("""sample=df.sample(n=20000, random_state=42).reset_index(drop=True)
X=pd.DataFrame([extract_all(u) for u in sample['url']])[ALL_COLS]
y=(sample['label']=='phishing').astype(int).values
print('X', X.shape, 'y', y.shape)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
scaler=StandardScaler()
Xtr_s=scaler.fit_transform(Xtr); Xte_s=scaler.transform(Xte)
print('split done')""")

md("## 3. Train models & evaluate")
code("""models={
 'LogisticRegression': LogisticRegression(max_iter=1000),
 'SVM (RBF)': SVC(kernel='rbf', probability=True),
 'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
 'XGBoost': XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss', n_jobs=-1),
}
results=[]
for name,m in models.items():
    if name in ('LogisticRegression','SVM (RBF)'):
        m.fit(Xtr_s,ytr); pred=m.predict(Xte_s); proba=m.predict_proba(Xte_s)[:,1]
    else:
        m.fit(Xtr,ytr); pred=m.predict(Xte); proba=m.predict_proba(Xte)[:,1]
    acc=accuracy_score(yte,pred); prec=precision_score(yte,pred)
    rec=recall_score(yte,pred); f1=f1_score(yte,pred)
    fpr,tpr,_=roc_curve(yte,proba); roc_auc=auc(fpr,tpr)
    results.append({'model':name,'accuracy':acc,'precision':prec,'recall':rec,'f1':f1,'auc':roc_auc,'fpr':fpr,'tpr':tpr})
    print(f'{name}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} auc={roc_auc:.4f}')
    ConfusionMatrixDisplay(confusion_matrix(yte,pred),display_labels=['legitimate','phishing']).plot(cmap='Blues')
    plt.title(name); plt.show()
pd.DataFrame(results)[['model','accuracy','precision','recall','f1','auc']]""")

md("## 4. ROC curves")
code("""fig,ax=plt.subplots(figsize=(7,6))
for r in results:
    ax.plot(r['fpr'],r['tpr'],label=f"{r['model']} (AUC={r['auc']:.3f})")
ax.plot([0,1],[0,1],'k--',label='Random')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves'); ax.legend(); plt.tight_layout(); plt.show()""")

md("## 5. Save best model (XGBoost) for predict.py")
code("""best=XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss', n_jobs=-1)
best.fit(X,y)
joblib.dump({'model':best,'features':ALL_COLS,'scaler':None},'model.pkl')
print('saved model.pkl (XGBoost on full sample)')""")

nb['cells']=cells
with open('/home/aaron/MTECH/aismp/full_pipeline.ipynb','w') as f:
    nbf.write(nb,f)
print('written')