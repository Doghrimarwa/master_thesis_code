"""
Stand-in for FARM's doc_classification.py (van Aken et al.), used ONLY by the
smoke test, because FARM needs a separate Python 3.8 environment.

It reads the same YAML config and split files and writes eval_results.txt in
the same format, but trains a TF-IDF + logistic regression model instead of
Bio_ClinicalBERT. Accepts the same command-line flags as the real script.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

p = argparse.ArgumentParser()
p.add_argument("--task_config", required=True)
p.add_argument("--seed", type=int, default=11)
a, _ = p.parse_known_args()
cfg = yaml.safe_load(open(a.task_config))
d = Path(cfg["data"]["data_dir"])
lab = cfg["data"]["parsing"]["label_column"]
tr = pd.read_csv(d / cfg["data"]["train_filename"])
te = pd.read_csv(d / cfg["data"]["test_filename"])
vec = TfidfVectorizer().fit(tr["text"])
Xtr, Xte = vec.transform(tr["text"]), vec.transform(te["text"])
if cfg["multilabel"]:
    labels = (d / cfg["data"]["label_list"]).read_text().split(" ")
    mlb = MultiLabelBinarizer(classes=labels)
    ytr = mlb.fit_transform(tr[lab].astype(str).str.split(","))
    yte = mlb.transform(te[lab].astype(str).str.split(","))
    clf = OneVsRestClassifier(LogisticRegression(max_iter=200, random_state=a.seed)).fit(Xtr, ytr)
    keep = (yte.sum(0) > 0) & (yte.sum(0) < len(yte))
    roc = roc_auc_score(yte[:, keep], clf.predict_proba(Xte)[:, keep], average="macro")
else:
    clf = LogisticRegression(max_iter=200, random_state=a.seed).fit(Xtr, tr[lab])
    roc = roc_auc_score(te[lab], clf.predict_proba(Xte)[:, 1]) if te[lab].nunique() > 1 else 0.5
out = Path(cfg["output_dir"]) / "stub_run"
out.mkdir(parents=True, exist_ok=True)
(out / "eval_results.txt").write_text(f"roc_auc: {roc:.4f}\nf1_macro: {np.nan}\n")
print(f"[stub] ROC-AUC {roc:.4f} -> {out/'eval_results.txt'}")
