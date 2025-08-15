from __future__ import annotations
import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from preprocess import HEART_FEATURES, generate_synthetic_heart, fit_scaler, transform

ROOT = Path(__file__).resolve().parents[2]  # project root
MODELS = ROOT / "models"
ENC = MODELS / "encoders_scalers"
MODELS.mkdir(parents=True, exist_ok=True)
ENC.mkdir(parents=True, exist_ok=True)

def main():
    # generate reproducible synthetic data
    df = generate_synthetic_heart(n=800, seed=7)
    X = df[HEART_FEATURES]
    y = df["target"]

    scaler = fit_scaler(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=123)
    X_tr_sc = transform(scaler, X_tr)
    X_te_sc = transform(scaler, X_te)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr_sc, y_tr)

    y_prob = clf.predict_proba(X_te_sc)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    auc = roc_auc_score(y_te, y_prob)
    acc = accuracy_score(y_te, y_pred)
    print(f"[Heart] AUC={auc:.3f} ACC={acc:.3f}")

    joblib.dump(clf, MODELS / "heart_model.pkl")
    joblib.dump(scaler, ENC / "heart_preproc.pkl")

if __name__ == "__main__":
    main()
