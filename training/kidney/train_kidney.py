from __future__ import annotations
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from preprocess import KIDNEY_FEATURES, generate_synthetic_kidney, fit_scaler

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
ENC = MODELS / "encoders_scalers"
MODELS.mkdir(exist_ok=True)
ENC.mkdir(parents=True, exist_ok=True)

def main():
    df = generate_synthetic_kidney(n=700, seed=99)
    X = df[KIDNEY_FEATURES]
    y = df["target"]

    scaler = fit_scaler(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=123)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(scaler.transform(X_tr), y_tr)

    y_prob = clf.predict_proba(scaler.transform(X_te))[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    auc = roc_auc_score(y_te, y_prob)
    acc = accuracy_score(y_te, y_pred)
    print(f"[Kidney] AUC={auc:.3f} ACC={acc:.3f}")

    joblib.dump(clf, MODELS / "kidney_model.pkl")
    joblib.dump(scaler, ENC / "kidney_preproc.pkl")

if __name__ == "__main__":
    main()
