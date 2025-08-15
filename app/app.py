from __future__ import annotations
import streamlit as st
from pathlib import Path
from components.ui import page_header, info

st.set_page_config(page_title="Multiple Disease Predictor (MVP-1)", page_icon="🩺", layout="centered")

page_header("Multiple Disease Prediction — MVP-1", "Heart predictor + Symptom Checker (alpha)")

st.write(
    """
    **MVP-1 scope:**  
    - ✅ Heart disease prediction (demo model)  
    - ✅ Symptom Checker (alpha, dropdown-based rule mapping)  
    - 🚧 Kidney/Liver/Lungs predictors will arrive in next MVPs  
    """
)

info("Use the left sidebar **Pages** to navigate: Heart / Symptom Checker.")
st.caption("Built in strict MVP style. Next iterations will add more diseases + better models.")
