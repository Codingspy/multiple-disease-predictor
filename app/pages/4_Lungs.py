import streamlit as st
import joblib
import numpy as np
from pathlib import Path
import json
from components.ui import page_header, good, warn, err
from components.form_helpers import number_input, select_input

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "lungs_model.pkl"
SCALER_PATH = ROOT / "models" / "encoders_scalers" / "lungs_preproc.pkl"

def load_lung_schema():
    schema_path = ROOT / "data" / "schemas" / "lungs.json"
    return json.loads(schema_path.read_text())

@st.cache_resource
def load_artifacts():
    clf = joblib.load(MODEL_PATH)
    sc = joblib.load(SCALER_PATH)
    return clf, sc

def render_form():
    schema = load_lung_schema()
    vals = {}
    st.subheader("Enter Lung-Related Details")
    for feat in schema["features"]:
        if "choices" in feat:
            vals[feat["name"]] = select_input(feat["name"], feat["choices"])
        else:
            vals[feat["name"]] = number_input(feat["name"], feat["min"], feat["max"], step=1, value=feat["min"])
    return vals

def main():
    page_header("Lung Disease Predictor", "Synthetic logistic model — MVP-4")
    try:
        clf, sc = load_artifacts()
        vals = render_form()
        if st.button("Predict"):
            order = [f["name"] for f in load_lung_schema()["features"]]
            x = np.array([[vals[k] for k in order]], dtype=float)
            x_sc = sc.transform(x)
            prob = float(clf.predict_proba(x_sc)[0,1])
            pred = "High Risk" if prob >= 0.5 else "Low Risk"
            good(f"Prediction: **{pred}**")
            st.write(f"Estimated probability: **{prob:.2%}**")
            if pred == "High Risk":
                warn("Consult a medical professional.")
    except Exception as e:
        err(str(e))

if __name__ == "__main__":
    main()
