from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import hashlib
from datetime import datetime, timezone

import streamlit as st

from app.core.config import get_paths
from app.core.evidence import build_evidence_map
from app.core.text_extract import extract_text
from app.core.resume_structurer import build_candidate_profile



def _safe_filename(name: str) -> str:
    # Keep it simple and safe for Windows/Cloud
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", ".", " "):
            keep.append(ch)
    cleaned = "".join(keep).strip().replace(" ", "_")
    return cleaned or "resume"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


st.set_page_config(page_title="Resume Intake", layout="wide")

paths = get_paths()

st.title("Resume Intake")
st.caption("Upload your resume. We will extract text and store a clean source for downstream matching.")

st.divider()

uploaded = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf", "docx", "txt"])

col_left, col_right = st.columns([1, 1], gap="large")

if uploaded is None:
    with col_left:
        st.info("Upload a file to begin.")
    st.stop()

file_bytes = uploaded.getvalue()
file_hash = _sha256_bytes(file_bytes)

now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
base_name = _safe_filename(Path(uploaded.name).stem)
ext = Path(uploaded.name).suffix.lower() or ".bin"

# Keep a stable version folder across Streamlit reruns for the same uploaded file
state_key = f"resume_version_dir__{file_hash[:16]}"

if state_key not in st.session_state:
    st.session_state[state_key] = f"{now}_{file_hash[:10]}"

version_dir = paths.resume / st.session_state[state_key]
version_dir.mkdir(parents=True, exist_ok=True)

original_path = version_dir / f"{base_name}{ext}"
text_path = version_dir / "resume_text.txt"
meta_path = version_dir / "meta.txt"
evidence_path = version_dir / "evidence_map.json"


def handle_save_and_extract(
    *,
    uploaded,
    file_bytes: bytes,
    original_path: Path,
    text_path: Path,
    meta_path: Path,
    file_hash: str,
    now: str,
    version_dir: Path,
) -> None:
    """
    Save the uploaded file, extract text to resume_text.txt, write meta.txt.
    UI side-effects: st.success / st.warning messages.
    """
    original_path.write_bytes(file_bytes)

    result = extract_text(original_path)
    text_path.write_text(result.text, encoding="utf-8")

    meta_lines = [
        f"uploaded_name={uploaded.name}",
        f"stored_original={original_path.name}",
        f"file_sha256={file_hash}",
        f"file_type={result.file_type}",
        f"saved_at_utc={now}",
        f"text_chars={len(result.text)}",
    ]
    meta_path.write_text("\n".join(meta_lines), encoding="utf-8")

    st.success(f"Saved to: {version_dir.as_posix()}")

    if result.warnings:
        st.warning("Extraction warnings:")
        for w in result.warnings:
            st.write(f"- {w}")


def handle_preview_and_evidence(
    *,
    file_bytes: bytes,
    original_path: Path,
    text_path: Path,
    version_dir: Path,
    base_name: str,
    ext: str,
    evidence_path: Path,
    max_preview_chars: int = 15000,
) -> None:
    """
    Show extracted text preview. If saved text exists, use it.
    Otherwise extract via temporary file for preview.
    Also shows evidence_map preview if present.
    """
    preview_text: str = ""

    # If user hasn’t clicked save yet, do a preview extraction in-memory:
    if original_path.exists() and text_path.exists():
        preview_text = text_path.read_text(encoding="utf-8", errors="ignore")
        st.caption("Showing saved extracted text.")
    else:
        # Temporary preview: write to a temp path inside version folder without committing original
        temp_path = version_dir / f"__temp__{base_name}{ext}"
        temp_path.write_bytes(file_bytes)
        result = extract_text(temp_path)
        preview_text = result.text or ""
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        st.caption("Preview extracted text (not saved yet). Click save to persist files.")

        if result.warnings:
            st.warning("Preview warnings:")
            for w in result.warnings:
                st.write(f"- {w}")

    st.text_area("Extracted resume text", value=preview_text[:max_preview_chars], height=520)
    st.caption("Preview is capped in the UI. Full text is saved in resume_text.txt after you click Save.")

    st.write("")
    st.subheader("Evidence preview")

    if evidence_path.exists():
        data = json.loads(evidence_path.read_text(encoding="utf-8", errors="ignore"))
        total = data.get("meta", {}).get("total_evidence", 0)
        st.caption(f"Evidence items: {total}")

        for item in data.get("evidence", [])[:20]:
            st.write(f"{item['id']} [{item['section']}] {item['source_text']}")
    else:
        st.caption("Generate evidence_map.json to preview evidence items here.")


with col_left:
    st.subheader("Stored files")

    if st.button("Save uploaded file + extract text", type="primary"):
        handle_save_and_extract(
            uploaded=uploaded,
            file_bytes=file_bytes,
            original_path=original_path,
            text_path=text_path,
            meta_path=meta_path,
            file_hash=file_hash,
            now=now,
            version_dir=version_dir,
        )

    if st.button("Generate evidence_map.json (no OpenAI)"):
        if not text_path.exists():
            st.error("resume_text.txt not found yet. Click 'Save uploaded file + extract text' first.")
        else:
            resume_text = text_path.read_text(encoding="utf-8", errors="ignore")
            evidence_map = build_evidence_map(resume_text)
            evidence_path.write_text(json.dumps(evidence_map, indent=2), encoding="utf-8")
            st.success(
                f"Generated: {evidence_path.name} ({evidence_map['meta']['total_evidence']} evidence items)"
            )
    profile_path = version_dir / "candidate_profile.json"

    if st.button("Generate candidate_profile.json (OpenAI strict)"):
        if not evidence_path.exists():
            st.error("evidence_map.json not found yet. Generate evidence_map first.")
        else:
            try:
                evidence_map = json.loads(evidence_path.read_text(encoding="utf-8", errors="ignore"))
                profile = build_candidate_profile(evidence_map=evidence_map, model="gpt-4o-mini")
                profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
                st.success(f"Generated: {profile_path.name}")
            except Exception as e:
                st.error(str(e))

    st.write("")
    st.code(str(version_dir), language="text")

with col_right:
    st.subheader("Preview")

    handle_preview_and_evidence(
        file_bytes=file_bytes,
        original_path=original_path,
        text_path=text_path,
        version_dir=version_dir,
        base_name=base_name,
        ext=ext,
        evidence_path=evidence_path,
        max_preview_chars=15000,
    )
