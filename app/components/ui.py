from __future__ import annotations
import streamlit as st

def page_header(title: str, subtitle: str | None = None):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()

def metric_row(items: list[tuple[str, str]]):
    cols = st.columns(len(items))
    for (c, (label, val)) in zip(cols, items):
        c.metric(label, val)

def info(msg: str): st.info(msg)
def warn(msg: str): st.warning(msg)
def err(msg: str): st.error(msg)
def good(msg: str): st.success(msg)
