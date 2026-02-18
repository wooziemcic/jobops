# app/pages/03_Ranking.py
from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.ranker import find_latest_resume_folder, load_json, load_jobs, rank_jobs
from app.core.openai_rerank_legacy import openai_rerank_topn_legacy

st.set_page_config(page_title="Ranking", layout="wide")

st.title("Ranking")
st.caption("Deterministic resume ↔ job match scoring + optional OpenAI rerank (legacy openai==0.28.x).")
st.divider()

# -----------------------------
# Paths
# -----------------------------
DATA = ROOT / "data"
RESUME_ROOT = DATA / "resume"
JOBS_LATEST = DATA / "jobs" / "normalized" / "jobs_latest.json"
RANKED_DIR = DATA / "jobs" / "ranked"
RANKED_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Locate latest resume artifacts
# -----------------------------
latest_resume_dir = find_latest_resume_folder(RESUME_ROOT)
if latest_resume_dir is None:
    st.error("No resume found under data/resume/. Upload a resume in Page 01 first.")
    st.stop()

prof_path = latest_resume_dir / "candidate_profile.json"
text_path = latest_resume_dir / "resume_text.txt"

if not prof_path.exists():
    st.error(f"Missing candidate_profile.json in: {latest_resume_dir}")
    st.stop()
if not text_path.exists():
    st.error(f"Missing resume_text.txt in: {latest_resume_dir}")
    st.stop()
if not JOBS_LATEST.exists():
    st.error("Missing jobs_latest.json. Run Page 02 Job Feed first.")
    st.stop()

# -----------------------------
# Sidebar controls (ALL controls go here)
# -----------------------------
with st.sidebar:
    st.header("Ranking controls")

    top_k = st.slider("Show top K jobs", min_value=10, max_value=200, value=50, step=10)

    st.subheader("Weights")
    w_skills = st.slider("Weight: Skills", 0.0, 1.0, 0.45, 0.05)
    w_title = st.slider("Weight: Title", 0.0, 1.0, 0.25, 0.05)
    w_text = st.slider("Weight: Resume↔Job text", 0.0, 1.0, 0.20, 0.05)
    w_penalty = st.slider("Weight: Seniority penalty", 0.0, 1.0, 0.10, 0.05)
    normalize_weights = st.checkbox("Normalize weights to sum=1 (recommended)", value=True)

    st.divider()
    st.subheader("OpenAI rerank (optional)")
    enable_llm = st.checkbox("Use OpenAI to rerank top N", value=False)
    llm_top_n = st.slider("Top N to rerank", min_value=5, max_value=80, value=25, step=5)
    llm_model = st.text_input("Model", value="gpt-3.5-turbo")
    st.caption("Requires OPENAI_API_KEY in Streamlit secrets (and openai==0.28.x installed).")

    run_btn = st.button("Run ranking", type="primary")

# -----------------------------
# Inputs detected (main page)
# -----------------------------
col1, col2 = st.columns([2, 1], gap="large")
with col1:
    st.subheader("Inputs detected")
    st.write(f"Latest resume: `{latest_resume_dir.name}`")
    st.write(f"Profile: `{prof_path.name}`")
    st.write(f"Resume text: `{text_path.name}`")
    st.write(f"Jobs: `{JOBS_LATEST}`")
with col2:
    st.subheader("Outputs")
    st.write(f"Ranked folder: `{RANKED_DIR}`")
    st.write("Will write:")
    st.write("- `jobs_ranked_<timestamp>.json`")
    st.write("- `jobs_ranked_latest.json`")

if not run_btn:
    st.stop()

# -----------------------------
# Load inputs
# -----------------------------
profile = load_json(prof_path)
resume_text = text_path.read_text(encoding="utf-8", errors="ignore")
jobs = load_jobs(JOBS_LATEST)

if not isinstance(jobs, list) or not jobs:
    st.error("jobs_latest.json loaded empty or invalid. Re-run Job Feed.")
    st.stop()

# -----------------------------
# Build weights
# -----------------------------
weights = {
    "skills": float(w_skills),
    "title": float(w_title),
    "resume_job_text": float(w_text),
    "penalty": float(w_penalty),
}
if normalize_weights:
    s = sum(weights.values())
    if s <= 0:
        weights = {"skills": 1.0, "title": 0.0, "resume_job_text": 0.0, "penalty": 0.0}
    else:
        weights = {k: v / s for k, v in weights.items()}

# -----------------------------
# Run deterministic ranking
# -----------------------------
with st.spinner("Running deterministic ranking..."):
    ranked = rank_jobs(jobs, profile, resume_text, weights=weights)

# Convert ranked objects -> JSON serializable rows
rows = []
for r in ranked:
    rows.append(
        {
            "score": float(getattr(r, "score", 0.0)),
            "breakdown": getattr(r, "breakdown", {}) or {},
            "matched_skills": getattr(r, "matched_skills", []) or [],
            "title_hits": getattr(r, "title_hits", []) or [],
            "overlap_tokens": getattr(r, "overlap_tokens", []) or [],
            "job": getattr(r, "job", {}) or {},
        }
    )

# -----------------------------
# Optional OpenAI rerank (legacy openai==0.28.x)
# -----------------------------
if enable_llm:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OPENAI_API_KEY not found in Streamlit secrets. Disable OpenAI rerank or add the key.")
    else:
        with st.spinner(f"OpenAI reranking top {llm_top_n}..."):
            try:
                rows = openai_rerank_topn_legacy(
                    candidate_profile=profile,
                    resume_text=resume_text,
                    ranked_rows=rows,
                    top_n=int(llm_top_n),
                    model=(llm_model.strip() or "gpt-3.5-turbo"),
                    api_key=api_key,
                )
            except Exception as e:
                st.error(f"OpenAI rerank failed: {e}")
                # ensure final_score exists even if rerank fails
                for row in rows:
                    row["llm"] = None
                    row["final_score"] = float(row.get("score") or 0.0)
else:
    for row in rows:
        row["llm"] = None
        row["final_score"] = float(row.get("score") or 0.0)

# -----------------------------
# Save outputs (timestamped + latest alias)
# -----------------------------
ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out_path = RANKED_DIR / f"jobs_ranked_{ts}.json"
latest_path = RANKED_DIR / "jobs_ranked_latest.json"

out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
latest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

st.success(f"Saved ranked output: `{out_path.name}` and `jobs_ranked_latest.json`")

# -----------------------------
# Display top results
# -----------------------------
st.divider()
st.subheader("Top matches")

show_rows = rows[: int(top_k)]

for i, row in enumerate(show_rows, start=1):
    j = row.get("job") if isinstance(row, dict) else {}
    if not isinstance(j, dict):
        j = {}

    title = (j.get("title") or "").strip()
    company = (j.get("company") or "").strip()
    loc = (j.get("location") or "").strip()
    apply_url = (j.get("apply_url") or "").strip()
    final_score = float(row.get("final_score") or 0.0)

    header = f"{i}. {title} — {company} ({loc}) | final={final_score:.3f}"
    with st.expander(header, expanded=(i <= 5)):
        c1, c2, c3 = st.columns([1, 1, 2], gap="large")

        with c1:
            st.write("Deterministic breakdown")
            st.json(row.get("breakdown") or {})

        with c2:
            st.write("Matched skills")
            ms = row.get("matched_skills") or []
            st.write(", ".join(ms[:20]) if ms else "None")

            st.write("Title hits")
            th = row.get("title_hits") or []
            st.write(", ".join(th[:15]) if th else "None")

        with c3:
            if apply_url.startswith("http"):
                st.markdown(f"[Apply here]({apply_url})")
            else:
                st.write("No apply link")

            llm = row.get("llm")
            if isinstance(llm, dict):
                st.write("OpenAI rerank")
                st.write(f"LLM score: **{llm.get('llm_score')} / 100**")

                reasons = llm.get("llm_reasons") or []
                if reasons:
                    st.write("Reasons:")
                    for rtxt in reasons[:5]:
                        st.write(f"- {rtxt}")

                gaps = llm.get("missing_gaps") or []
                if gaps:
                    st.write("Gaps:")
                    for g in gaps[:8]:
                        st.write(f"- {g}")

            desc = (j.get("description") or "").strip()
            if desc:
                st.write("Job description (preview)")
                st.write(desc[:900] + ("..." if len(desc) > 900 else ""))
