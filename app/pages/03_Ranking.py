# app/pages/03_Ranking.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
from datetime import datetime, timezone

import streamlit as st

from app.core.ranker import (
    find_latest_resume_folder,
    load_json,
    load_jobs,
    rank_jobs,
)

st.set_page_config(page_title="Ranking", layout="wide")
st.title("Ranking")
st.caption("Deterministic resume ↔ job match scoring (no OpenAI). Auditable score breakdown per job.")
st.divider()

DATA = ROOT / "data"
RESUME_ROOT = DATA / "resume"
JOBS_LATEST = DATA / "jobs" / "normalized" / "jobs_latest.json"
RANKED_OUT = DATA / "jobs" / "ranked"
RANKED_OUT.mkdir(parents=True, exist_ok=True)

latest_resume_dir = find_latest_resume_folder(RESUME_ROOT)

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Inputs detected")

    if latest_resume_dir is None:
        st.error("No resume folder found under data/resume/. Upload a resume in Page 01 first.")
        st.stop()

    prof_path = latest_resume_dir / "candidate_profile.json"
    text_path = latest_resume_dir / "resume_text.txt"

    if not prof_path.exists() or not text_path.exists():
        st.error(f"Missing candidate_profile.json or resume_text.txt in: {latest_resume_dir}")
        st.stop()

    st.write(f"Latest resume: `{latest_resume_dir.name}`")
    st.write(f"Profile: `{prof_path.name}`")
    st.write(f"Resume text: `{text_path.name}`")

    if not JOBS_LATEST.exists():
        st.error("Missing jobs_latest.json. Run Page 02 Job Feed first.")
        st.stop()

    st.write(f"Jobs: `{JOBS_LATEST}`")

with col2:
    st.subheader("Ranking controls")

    top_k = st.slider("Show top K jobs", min_value=10, max_value=200, value=50, step=10)

    w_skills = st.slider("Weight: Skills", 0.0, 1.0, 0.45, 0.05)
    w_title = st.slider("Weight: Title", 0.0, 1.0, 0.25, 0.05)
    w_text = st.slider("Weight: Resume↔Job text", 0.0, 1.0, 0.20, 0.05)
    w_pen = st.slider("Weight: Seniority penalty", 0.0, 1.0, 0.10, 0.05)

    normalize_weights = st.checkbox("Normalize weights to sum=1 (recommended)", value=True)

    run = st.button("Run ranking", type="primary")

if run:
    prof = load_json(prof_path)
    resume_text = text_path.read_text(encoding="utf-8", errors="ignore")
    jobs = load_jobs(JOBS_LATEST)

    weights = {"skills": w_skills, "title": w_title, "resume_job_text": w_text, "penalty": w_pen}

    if normalize_weights:
        s = max(1e-9, (weights["skills"] + weights["title"] + weights["resume_job_text"] + weights["penalty"]))
        weights = {k: v / s for k, v in weights.items()}

    ranked = rank_jobs(jobs, prof, resume_text, weights=weights)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RANKED_OUT / f"jobs_ranked_{ts}.json"

    payload = []
    for r in ranked:
        payload.append(
            {
                "score": r.score,
                "breakdown": r.breakdown,
                "matched_skills": r.matched_skills,
                "title_hits": r.title_hits,
                "overlap_tokens": r.overlap_tokens,
                "job": r.job,
            }
        )

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    st.success(f"Saved ranked output: {out_path.name}")

    st.divider()
    st.subheader("Top matches")

    for i, r in enumerate(ranked[:top_k], start=1):
        j = r.job
        title = j.get("title") or ""
        company = j.get("company") or ""
        loc = j.get("location") or ""
        apply_url = j.get("apply_url") or ""

        header = f"{i}. {title} — {company} ({loc})  |  score={r.score:.3f}"
        with st.expander(header, expanded=(i <= 5)):
            cols = st.columns([1, 1, 2])
            with cols[0]:
                st.write("Breakdown")
                st.json(r.breakdown)
            with cols[1]:
                st.write("Matched skills")
                st.write(", ".join(r.matched_skills[:20]) if r.matched_skills else "None")
                st.write("Title hits")
                st.write(", ".join(r.title_hits) if r.title_hits else "None")
            with cols[2]:
                if apply_url.startswith("http"):
                    st.markdown(f"[Apply here]({apply_url})")
                st.write("Evidence tokens (resume↔job overlap)")
                st.write(", ".join(r.overlap_tokens[:30]) if r.overlap_tokens else "None")

                desc = (j.get("description") or "").strip()
                if desc:
                    st.write("Job description (preview)")
                    st.write(desc[:900] + ("..." if len(desc) > 900 else ""))
