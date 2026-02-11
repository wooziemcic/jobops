from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path so "import app...." works in Streamlit
ROOT = Path(__file__).resolve().parents[1]  # jobops/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="JobOps",
    layout="wide",
)

st.title("JobOps")
st.caption("Resume Intake → Job Feed → Ranking → Application Kit → History")

st.divider()
st.write("Baseline app is running.")
