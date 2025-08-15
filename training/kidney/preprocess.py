from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

KIDNEY_FEATURES = [
    "age","bp","sg","al","su","bgr","bu","sc",
    "sod","pot","hemo","pcv","wbcc","rbcc"
]

def generate_synthetic_kidney(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(2, 90, size=n),
        "bp": rng.integers(50, 180, size=n),
        "sg": rng.uniform(1.005, 1.025, size=n).round(3),
        "al": rng.integers(0, 6, size=n),
        "su": rng.integers(0, 6, size=n),
        "bgr": rng.integers(50, 500, size=n),
        "bu": rng.integers(1, 150, size=n),
        "sc": rng.uniform(0.4, 15.0, size=n).round(2),
        "sod": rng.integers(100, 170, size=n),
        "pot": rng.uniform(2.5, 7.0, size=n).round(2),
        "hemo": rng.uniform(3.1, 17.8, size=n).round(1),
        "pcv": rng.integers(20, 54, size=n),
        "wbcc": rng.integers(3000, 12000, size=n),
        "rbcc": rng.uniform(2.5, 6.5, size=n).round(1)
    })
    # Label: high bp, low sg, high sc, low hemo more likely disease
    score = (
        0.02*df["bp"] - 10*(df["sg"] - 1.010) + 0.3*df["al"] + 0.3*df["su"] +
        0.01*df["bgr"] + 0.01*df["bu"] + 0.5*df["sc"] -
        0.1*df["hemo"] - 0.05*df["pcv"]
    )
    p = 1 / (1 + np.exp(-(score - score.mean()) / (score.std() + 1e-6)))
    y = (p > 0.5).astype(int)
    df["target"] = y
    return df

def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    sc = StandardScaler()
    sc.fit(X)
    return sc
