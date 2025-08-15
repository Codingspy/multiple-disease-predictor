from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

LIVER_FEATURES = [
    "age","gender","total_bilirubin","direct_bilirubin",
    "alkaline_phosphotase","alanine_aminotransferase","aspartate_aminotransferase",
    "total_proteins","albumin","albumin_globulin_ratio"
]

def generate_synthetic_liver(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(4, 90, size=n),
        "gender": rng.integers(0, 2, size=n),
        "total_bilirubin": rng.uniform(0.1, 75.0, size=n).round(2),
        "direct_bilirubin": rng.uniform(0.0, 20.0, size=n).round(2),
        "alkaline_phosphotase": rng.integers(50, 3000, size=n),
        "alanine_aminotransferase": rng.integers(0, 1000, size=n),
        "aspartate_aminotransferase": rng.integers(0, 1000, size=n),
        "total_proteins": rng.uniform(2.0, 9.5, size=n).round(1),
        "albumin": rng.uniform(0.5, 6.0, size=n).round(1),
        "albumin_globulin_ratio": rng.uniform(0.1, 3.0, size=n).round(2)
    })
    # Higher bilirubin, high enzyme levels, low albumin → disease
    score = (
        0.05*df["total_bilirubin"] + 0.03*df["direct_bilirubin"] +
        0.001*df["alkaline_phosphotase"] + 0.002*df["alanine_aminotransferase"] +
        0.002*df["aspartate_aminotransferase"] - 0.2*df["albumin"]
    )
    p = 1 / (1 + np.exp(-(score - score.mean()) / (score.std() + 1e-6)))
    y = (p > 0.5).astype(int)
    df["target"] = y
    return df

def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    sc = StandardScaler()
    sc.fit(X)
    return sc
