# app/pages/04_Application_Kit.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
from datetime import datetime, timezone

import streamlit as st

from app.core.application_kit import save_kit
from app.core.ranker import find_latest_resume_folder, load_json, load_jobs

st.set_page_config(page_title="Application Kit", layout="wide")
st.title("Application Kit")
st.caption("Generate a clean, auditable application kit for a selected job (deterministic baseline, no OpenAI).")
st.divider()

DATA = ROOT / "data"
RESUME_ROOT = DATA / "resume"
JOBS_LATEST = DATA / "jobs" / "normalized" / "jobs_latest.json"
RANKED_DIR = DATA / "jobs" / "ranked"
KITS_ROOT = DATA / "kits"
KITS_ROOT.mkdir(parents=True, exist_ok=True)

latest_resume_dir = find_latest_resume_folder(RESUME_ROOT)
if latest_resume_dir is None:
    st.error("No resume found under data/resume/. Upload a resume in Page 01 first.")
    st.stop()

prof_path = latest_resume_dir / "candidate_profile.json"
text_path = latest_resume_dir / "resume_text.txt"

if not prof_path.exists() or not text_path.exists():
    st.error(f"Missing candidate_profile.json or resume_text.txt in: {latest_resume_dir}")
    st.stop()

profile = load_json(prof_path)
resume_text = text_path.read_text(encoding="utf-8", errors="ignore")

st.subheader("Inputs")
st.write(f"Latest resume folder: `{latest_resume_dir.name}`")

# Prefer ranked output if available, else fall back to jobs_latest.json
ranked_files = sorted([p for p in RANKED_DIR.glob("jobs_ranked_*.json") if p.is_file()], reverse=True)
use_ranked = st.checkbox("Use latest ranked output (recommended)", value=bool(ranked_files))

jobs = []
if use_ranked and ranked_files:
    ranked_path = ranked_files[0]
    st.write(f"Using ranked file: `{ranked_path.name}`")
    ranked_payload = json.loads(ranked_path.read_text(encoding="utf-8", errors="ignore"))
    # ranked payload rows: {"score":..., "job": {...}, ...}
    jobs = [row.get("job") for row in ranked_payload if isinstance(row, dict) and isinstance(row.get("job"), dict)]
else:
    if not JOBS_LATEST.exists():
        st.error("Missing jobs_latest.json. Run Page 02 Job Feed first.")
        st.stop()
    st.write(f"Using jobs file: `{JOBS_LATEST}`")
    jobs = load_jobs(JOBS_LATEST)

if not jobs:
    st.error("No jobs available to build a kit. Run Job Feed (and Ranking if enabled) first.")
    st.stop()

st.divider()
st.subheader("Select a job")

# Build a selector label list
labels = []
for idx, j in enumerate(jobs):
    title = (j.get("title") or "").strip()
    company = (j.get("company") or "").strip()
    loc = (j.get("location") or "").strip()
    labels.append(f"{idx+1}. {title} — {company} ({loc})")

selected = st.selectbox("Job", options=list(range(len(jobs))), format_func=lambda i: labels[i])

job = jobs[selected]

# Preview
col1, col2 = st.columns([2, 1], gap="large")
with col1:
    st.write("Job preview")
    st.write(f"Title: {job.get('title','')}")
    st.write(f"Company: {job.get('company','')}")
    st.write(f"Location: {job.get('location','')}")
    if (job.get("apply_url") or "").startswith("http"):
        st.markdown(f"[Apply link]({job.get('apply_url')})")
with col2:
    st.write("Description preview")
    desc = (job.get("description") or "").strip()
    st.write(desc[:800] + ("..." if len(desc) > 800 else ""))

st.divider()

kit_name_hint = st.text_input("Optional kit note (added to manifest only)", value="")
create_btn = st.button("Create Application Kit", type="primary")

if create_btn:
    kit_dir = save_kit(KITS_ROOT, job, profile, resume_text)

    # If user added a note, append into manifest
    if kit_name_hint.strip():
        mpath = kit_dir / "manifest.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        manifest["note"] = kit_name_hint.strip()
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    st.success(f"Kit created: {kit_dir.name}")

    st.subheader("Artifacts")
    for fname in [
        "match_report.md",
        "recruiter_email.txt",
        "linkedin_dm.txt",
        "resume_tweak_suggestions.md",
        "job_snapshot.json",
        "profile_snapshot.json",
        "manifest.json",
    ]:
        p = kit_dir / fname
        if p.exists():
            st.write(f"- `{fname}`")

    # Quick render for convenience
    st.divider()
    st.subheader("Recruiter email draft")
    st.code((kit_dir / "recruiter_email.txt").read_text(encoding="utf-8"), language="text")

    st.subheader("LinkedIn DM draft")
    st.code((kit_dir / "linkedin_dm.txt").read_text(encoding="utf-8"), language="text")

    st.subheader("Resume tweak suggestions")
    st.markdown((kit_dir / "resume_tweak_suggestions.md").read_text(encoding="utf-8"))
