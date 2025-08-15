import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

LUNG_FEATURES = [
    "age", "gender", "smoking", "yellow_fingers", "anxiety",
    "peer_pressure", "chronic_disease", "fatigue", "allergy", "wheezing",
    "coughing", "shortness_of_breath", "swallowing_difficulty", "chest_pain"
]

def generate_synthetic_lungs(n=500, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(1, 90, size=n),
        "gender": rng.integers(0, 2, size=n),
        "smoking": rng.integers(0, 2, size=n),
        "yellow_fingers": rng.integers(0, 2, size=n),
        "anxiety": rng.integers(0, 2, size=n),
        "peer_pressure": rng.integers(0, 2, size=n),
        "chronic_disease": rng.integers(0, 2, size=n),
        "fatigue": rng.integers(0, 2, size=n),
        "allergy": rng.integers(0, 2, size=n),
        "wheezing": rng.integers(0, 2, size=n),
        "coughing": rng.integers(0, 2, size=n),
        "shortness_of_breath": rng.integers(0, 2, size=n),
        "swallowing_difficulty": rng.integers(0, 2, size=n),
        "chest_pain": rng.integers(0, 2, size=n)
    })
    score = (
        1.2*df["smoking"] + 0.8*df["yellow_fingers"] +
        0.9*df["shortness_of_breath"] + 0.7*df["chest_pain"] +
        0.5*df["coughing"]
    )
    prob = 1 / (1 + np.exp(-(score - score.mean()) / (score.std() + 1e-6)))
    df["target"] = (prob > 0.5).astype(int)
    return df

def fit_scaler(X):
    sc = StandardScaler()
    sc.fit(X)
    return sc
