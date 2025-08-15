from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HEART_FEATURES = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]

def generate_synthetic_heart(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(29, 77, size=n),
        "sex": rng.integers(0, 2, size=n),
        "cp": rng.integers(0, 4, size=n),
        "trestbps": rng.integers(90, 200, size=n),
        "chol": rng.integers(120, 565, size=n),
        "fbs": rng.integers(0, 2, size=n),
        "restecg": rng.integers(0, 2, size=n),
        "thalach": rng.integers(70, 210, size=n),
        "exang": rng.integers(0, 2, size=n),
        "oldpeak": rng.normal(1.0, 1.0, size=n).clip(0, 6),
        "slope": rng.integers(0, 3, size=n),
        "ca": rng.integers(0, 4, size=n),
        "thal": rng.integers(0, 3, size=n),
    })
    # simple rule to create a label: higher risk with age, chol, trestbps, exang, oldpeak
    score = (
        0.03*df["age"] + 0.01*df["chol"] + 0.02*df["trestbps"] +
        0.8*df["exang"] + 0.6*df["oldpeak"] - 0.01*df["thalach"]
    )
    p = 1 / (1 + np.exp(-(score - score.mean()) / (score.std() + 1e-6)))
    y = (p > 0.5).astype(int)
    df["target"] = y
    return df

def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    sc = StandardScaler()
    sc.fit(X)
    return sc

def transform(sc: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    return sc.transform(X[HEART_FEATURES])
