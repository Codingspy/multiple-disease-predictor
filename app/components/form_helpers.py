from __future__ import annotations
import json
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

def load_heart_schema():
    schema_path = ROOT / "data" / "schemas" / "heart.json"
    return json.loads(schema_path.read_text())

def number_input(label: str, min_value=None, max_value=None, step=1, value=None):
    return st.number_input(label, min_value=min_value, max_value=max_value, step=step, value=value)

def select_input(label: str, options: list[int], index: int = 0):
    return st.selectbox(label, options=options, index=index)
