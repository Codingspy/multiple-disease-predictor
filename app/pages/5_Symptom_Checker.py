from __future__ import annotations
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from components.ui import page_header, info

ROOT = Path(__file__).resolve().parents[2]
SYMPTOM_CSV = ROOT / "data" / "symptom_map.csv"

@st.cache_data
def load_map():
    df = pd.read_csv(SYMPTOM_CSV)
    return df

def score_conditions(symptoms: list[str], df: pd.DataFrame):
    if not symptoms:
        return {"heart":0,"kidney":0,"liver":0,"lungs":0}
    sub = df[df["symptom"].isin(symptoms)]
    scores = {cond: float(sub[cond].sum()) for cond in ["heart","kidney","liver","lungs"]}
    # normalize
    total = sum(scores.values()) or 1.0
    return {k: v/total for k,v in scores.items()}

def main():
    page_header("Symptom Checker (Alpha)", "Lightweight rule-based scorer")
    df = load_map()
    options = df["symptom"].tolist()
    sel = st.multiselect("Select your symptoms", options)
    if st.button("Check Possible Conditions"):
        scores = score_conditions(sel, df)
        st.write("**Relative Likelihood (normalized):**")
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        st.success(f"Top 2 likely conditions: {ranked[0][0].title()} ({ranked[0][1]:.0%}), {ranked[1][0].title()} ({ranked[1][1]:.0%})")

        fig, ax = plt.subplots()
        ax.bar(scores.keys(), scores.values())
        ax.set_ylabel("Likelihood")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        info("This is an MVP heuristic; not medical advice.")

if __name__ == "__main__":
    main()
