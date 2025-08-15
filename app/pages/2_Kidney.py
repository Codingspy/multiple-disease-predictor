from __future__ import annotations
import streamlit as st
import joblib
import numpy as np
from pathlib import Path
from components.ui import page_header, good, warn, err
from components.form_helpers import number_input, select_input
import json

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "kidney_model.pkl"
SCALER_PATH = ROOT / "models" / "encoders_scalers" / "kidney_preproc.pkl"

def load_kidney_schema():
    schema_path = ROOT / "data" / "schemas" / "kidney.json"
    return json.loads(schema_path.read_text())

@st.cache_resource
def load_artifacts():
    clf = joblib.load(MODEL_PATH)
    sc = joblib.load(SCALER_PATH)
    return clf, sc

def render_form():
    schema = load_kidney_schema()
    vals = {}
    st.subheader("Enter Kidney-Related Details")
    for feat in schema["features"]:
        name = feat["name"]
        if "choices" in feat:
            vals[name] = select_input(name, feat["choices"])
        else:
            step = 1 if feat["type"] == "int" else 0.01
            vals[name] = number_input(name, feat.get("min"), feat.get("max"), step=step, value=feat.get("min"))
    return vals

def main():
    page_header("Kidney Disease Predictor", "Synthetic model — MVP-2")
    try:
        clf, sc = load_artifacts()
        vals = render_form()
        if st.button("Predict"):
            order = [f["name"] for f in load_kidney_schema()["features"]]
            x = np.array([[vals[k] for k in order]], dtype=float)
            x_sc = sc.transform(x)
            prob = float(clf.predict_proba(x_sc)[0,1])
            pred = "High Risk" if prob >= 0.5 else "Low Risk"
            good(f"Prediction: **{pred}**")
            st.write(f"Estimated probability: **{prob:.2%}**")
            if pred == "High Risk":
                warn("Consult a clinician for further tests.")
    except Exception as e:
        err(str(e))

if __name__ == "__main__":
    main()
