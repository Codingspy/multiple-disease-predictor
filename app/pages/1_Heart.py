from __future__ import annotations
import streamlit as st
import joblib
import numpy as np
from pathlib import Path
from components.ui import page_header, good, warn, err
from components.form_helpers import load_heart_schema, number_input, select_input

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "heart_model.pkl"
SCALER_PATH = ROOT / "models" / "encoders_scalers" / "heart_preproc.pkl"

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Model artifacts missing. Please run training/heart/train_heart.py")
    clf = joblib.load(MODEL_PATH)
    sc = joblib.load(SCALER_PATH)
    return clf, sc

def render_form():
    schema = load_heart_schema()
    vals = {}
    st.subheader("Enter Details")
    for feat in schema["features"]:
        name = feat["name"]
        typ = feat["type"]
        if "choices" in feat:
            vals[name] = select_input(name, feat["choices"])
        else:
            if typ == "int":
                vals[name] = number_input(name, feat.get("min"), feat.get("max"), step=1, value=feat.get("min", 0))
            else:
                vals[name] = st.number_input(name, min_value=float(feat.get("min", 0.0)),
                                             max_value=float(feat.get("max", 10.0)),
                                             step=0.1, value=float(feat.get("min", 0.0)))
    return vals

def main():
    page_header("Heart Disease Predictor", "Demo logistic model (MVP-1)")
    try:
        clf, sc = load_artifacts()
        vals = render_form()
        if st.button("Predict"):
            order = [f["name"] for f in load_heart_schema()["features"]]
            x = np.array([[vals[k] for k in order]], dtype=float)
            x_sc = sc.transform(x)
            prob = float(clf.predict_proba(x_sc)[0,1])
            pred = "High Risk" if prob >= 0.5 else "Low Risk"
            good(f"Prediction: **{pred}**")
            st.write(f"Estimated probability of heart disease: **{prob:.2%}**")
            if pred == "High Risk":
                warn("This is a demo model; consult a qualified clinician for medical advice.")
    except Exception as e:
        err(str(e))

if __name__ == "__main__":
    main()
