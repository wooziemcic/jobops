# app/pages/04_Application_Kit.py
from __future__ import annotations

import sys
import json
import re
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.ranker import find_latest_resume_folder, load_json
from app.core.openai_kit_legacy import generate_application_kit_legacy

st.set_page_config(page_title="Application Kit", layout="wide")

st.title("Application Kit")
st.caption("Generate job-tailored kit from ranked jobs + resume/profile (OpenAI legacy openai==0.28.x).")
st.divider()

DATA = ROOT / "data"
RESUME_ROOT = DATA / "resume"
RANKED_LATEST = DATA / "jobs" / "ranked" / "jobs_ranked_latest.json"
KITS_DIR = DATA / "kits"
KITS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def _safe_slug(s: str, max_len: int = 80) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s[:max_len] if len(s) > max_len else s

def _load_ranked_rows() -> list[dict]:
    if not RANKED_LATEST.exists():
        return []
    try:
        data = json.loads(_read_text(RANKED_LATEST))
        return data if isinstance(data, list) else []
    except Exception:
        return []

# -----------------------------
# Locate latest resume artifacts
# -----------------------------
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
resume_text = _read_text(text_path)

ranked_rows = _load_ranked_rows()
if not ranked_rows:
    st.error("No ranked jobs found. Run Page 03 Ranking first to create jobs_ranked_latest.json.")
    st.stop()

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Kit controls")

    show_top_k = st.slider("Browse top K ranked jobs", 10, 200, 50, 10)

    # model controls (legacy)
    enable_openai = st.checkbox("Use OpenAI to generate kit", value=True)
    openai_model = st.text_input("Model", value="gpt-3.5-turbo")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    st.caption("Requires OPENAI_API_KEY in Streamlit secrets.")

    st.divider()
    generate_btn = st.button("Generate kit", type="primary")

# -----------------------------
# Main: choose a job
# -----------------------------
st.subheader("Inputs detected")
c1, c2 = st.columns([2, 1], gap="large")
with c1:
    st.write(f"Latest resume: `{latest_resume_dir.name}`")
    st.write(f"Profile: `{prof_path.name}`")
    st.write(f"Resume text: `{text_path.name}`")
    st.write(f"Ranked jobs: `{RANKED_LATEST}`")
with c2:
    st.write(f"Kits folder: `{KITS_DIR}`")

st.divider()
st.subheader("Select a job from ranked results")

top_rows = ranked_rows[: int(show_top_k)]

def _label_for_row(row: dict, idx: int) -> str:
    job = row.get("job") if isinstance(row, dict) else {}
    if not isinstance(job, dict):
        job = {}
    title = (job.get("title") or "Untitled").strip()
    company = (job.get("company") or "Unknown").strip()
    loc = (job.get("location") or "").strip()
    fs = row.get("final_score")
    if fs is None:
        fs = row.get("score", 0.0)
    try:
        fsf = float(fs)
    except Exception:
        fsf = 0.0
    return f"{idx:02d}. {title} — {company} ({loc}) | {fsf:.3f}"

options = { _label_for_row(r, i+1): r for i, r in enumerate(top_rows) }

selected_label = st.selectbox("Ranked jobs", list(options.keys()))
selected_row = options[selected_label]
selected_job = selected_row.get("job") if isinstance(selected_row, dict) else {}
if not isinstance(selected_job, dict):
    selected_job = {}

# Preview
st.markdown("### Job preview")
pc1, pc2 = st.columns([2, 1], gap="large")
with pc1:
    st.write(f"**Title:** {selected_job.get('title','')}")
    st.write(f"**Company:** {selected_job.get('company','')}")
    st.write(f"**Location:** {selected_job.get('location','')}")
    desc = (selected_job.get("description") or "").strip()
    if desc:
        st.write("**Description (preview):**")
        st.write(desc[:1200] + ("..." if len(desc) > 1200 else ""))
with pc2:
    st.write("**Scores**")
    st.json({
        "final_score": selected_row.get("final_score"),
        "base_score": selected_row.get("score"),
        "breakdown": selected_row.get("breakdown"),
    })
    apply_url = (selected_job.get("apply_url") or "").strip()
    if apply_url.startswith("http"):
        st.markdown(f"[Apply link]({apply_url})")

# -----------------------------
# Generate kit
# -----------------------------
if not generate_btn:
    st.stop()

# Validate OpenAI key if enabled
api_key = st.secrets.get("OPENAI_API_KEY", "")
if enable_openai and not api_key:
    st.error("OPENAI_API_KEY not found in Streamlit secrets. Add it or disable OpenAI generation.")
    st.stop()

# Create kit folder
ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
company_slug = _safe_slug(selected_job.get("company") or "Company")
title_slug = _safe_slug(selected_job.get("title") or "Role", max_len=95)
kit_dir = KITS_DIR / f"{ts}__{company_slug}__{title_slug}"
kit_dir.mkdir(parents=True, exist_ok=True)

# Always snapshot inputs for auditability
(kit_dir / "job_snapshot.json").write_text(json.dumps(selected_job, indent=2), encoding="utf-8")
(kit_dir / "profile_snapshot.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

manifest = {
    "created_at_utc": ts,
    "resume_folder": latest_resume_dir.name,
    "ranked_source": str(RANKED_LATEST),
    "job": {
        "title": selected_job.get("title"),
        "company": selected_job.get("company"),
        "location": selected_job.get("location"),
        "apply_url": selected_job.get("apply_url"),
        "source": selected_job.get("source"),
        "posted_at": selected_job.get("posted_at") or selected_job.get("date"),
    },
    "ranking": {
        "final_score": selected_row.get("final_score"),
        "base_score": selected_row.get("score"),
        "breakdown": selected_row.get("breakdown"),
        "matched_skills": selected_row.get("matched_skills"),
        "title_hits": selected_row.get("title_hits"),
    },
    "openai": {
        "enabled": bool(enable_openai),
        "model": openai_model,
        "temperature": float(temperature),
    },
    "outputs": {},
}

# Generate content
if enable_openai:
    with st.spinner("Generating tailored kit with OpenAI..."):
        kit = generate_application_kit_legacy(
            candidate_profile=profile,
            resume_text=resume_text,
            job=selected_job,
            ranking_row=selected_row,
            model=(openai_model.strip() or "gpt-3.5-turbo"),
            api_key=api_key,
            temperature=float(temperature),
        )

    # Write outputs
    (kit_dir / "match_report.md").write_text(kit["match_report_md"], encoding="utf-8")
    (kit_dir / "resume_tweak_suggestions.md").write_text(kit["resume_tweak_suggestions_md"], encoding="utf-8")
    (kit_dir / "recruiter_email.txt").write_text(kit["recruiter_email_txt"], encoding="utf-8")
    (kit_dir / "linkedin_dm.txt").write_text(kit["linkedin_dm_txt"], encoding="utf-8")

    manifest["outputs"] = {
        "match_report_md": "match_report.md",
        "resume_tweak_suggestions_md": "resume_tweak_suggestions.md",
        "recruiter_email_txt": "recruiter_email.txt",
        "linkedin_dm_txt": "linkedin_dm.txt",
    }
    manifest["kit_metadata"] = {
        "one_line_pitch": kit.get("one_line_pitch"),
        "keywords_to_emphasize": kit.get("keywords_to_emphasize"),
        "must_have_hits": kit.get("must_have_hits"),
        "gaps": kit.get("gaps"),
        "risk_flags": kit.get("risk_flags"),
    }
else:
    # If OpenAI disabled, still generate a minimal stub kit deterministically
    (kit_dir / "match_report.md").write_text("# Match Report\n(OpenAI disabled)\n", encoding="utf-8")
    (kit_dir / "resume_tweak_suggestions.md").write_text("# Resume Tweaks\n(OpenAI disabled)\n", encoding="utf-8")
    (kit_dir / "recruiter_email.txt").write_text("Hi,\n\n(OpenAI disabled)\n", encoding="utf-8")
    (kit_dir / "linkedin_dm.txt").write_text("Hi,\n\n(OpenAI disabled)\n", encoding="utf-8")
    manifest["outputs"] = {
        "match_report_md": "match_report.md",
        "resume_tweak_suggestions_md": "resume_tweak_suggestions.md",
        "recruiter_email_txt": "recruiter_email.txt",
        "linkedin_dm_txt": "linkedin_dm.txt",
    }

# Save manifest
(kit_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

st.success(f"Kit generated: `{kit_dir.name}`")

# -----------------------------
# Display outputs in UI
# -----------------------------
st.divider()
st.subheader("Generated kit")

tabs = st.tabs(["Match report", "Resume tweaks", "Recruiter email", "LinkedIn DM", "Manifest"])

with tabs[0]:
    st.markdown(_read_text(kit_dir / "match_report.md") or "")

with tabs[1]:
    st.markdown(_read_text(kit_dir / "resume_tweak_suggestions.md") or "")

with tabs[2]:
    st.text(_read_text(kit_dir / "recruiter_email.txt") or "")

with tabs[3]:
    st.text(_read_text(kit_dir / "linkedin_dm.txt") or "")

with tabs[4]:
    st.json(json.loads(_read_text(kit_dir / "manifest.json")))
