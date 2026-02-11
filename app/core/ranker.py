# app/core/ranker.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

TOKEN_RE = re.compile(r"[a-zA-Z0-9+#\.]+")

SENIOR_TOKENS = {
    "senior", "sr", "staff", "lead", "principal", "manager", "director", "vp", "head"
}
PHD_TOKENS = {"phd", "postdoc", "doctorate"}

DEFAULT_WEIGHTS = {
    "skills": 0.45,
    "title": 0.25,
    "resume_job_text": 0.20,
    "penalty": 0.10,
}

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def _tokens(s: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(s or "")]

def _token_set(s: str) -> set:
    return set(_tokens(s))

def _safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def extract_profile_signals(profile: Dict[str, Any], resume_text: str) -> Dict[str, Any]:
    """
    Tries to be schema-agnostic. Works even if your profile keys differ slightly.
    """
    # Skills can be in different places depending on your schema
    skills = (
        _safe_get(profile, ["skills"], [])
        or _safe_get(profile, ["candidate", "skills"], [])
        or _safe_get(profile, ["profile", "skills"], [])
        or []
    )
    if isinstance(skills, dict):
        # sometimes skills grouped; flatten values
        flat = []
        for v in skills.values():
            if isinstance(v, list):
                flat.extend(v)
        skills = flat

    skills = sorted({str(s).strip() for s in skills if str(s).strip()})
    skills_lc = sorted({_norm(s) for s in skills})

    # Target roles/titles
    target_titles = (
        _safe_get(profile, ["target_titles"], [])
        or _safe_get(profile, ["candidate", "target_titles"], [])
        or _safe_get(profile, ["profile", "target_titles"], [])
        or _safe_get(profile, ["experience", "target_titles"], [])
        or []
    )
    if isinstance(target_titles, str):
        target_titles = [target_titles]
    target_titles = [t.strip() for t in target_titles if str(t).strip()]

    # Candidate location or preferences (optional)
    location_pref = (
        _safe_get(profile, ["location_preferences"], None)
        or _safe_get(profile, ["candidate", "location_preferences"], None)
        or None
    )

    resume_tokens = _token_set(resume_text)

    return {
        "skills": skills,
        "skills_lc": skills_lc,
        "target_titles": target_titles,
        "location_preferences": location_pref,
        "resume_tokens": resume_tokens,
    }

def seniority_penalty(title: str, description: str) -> float:
    """
    Returns penalty score in [0,1] where 1 means heavy penalty.
    """
    tset = _token_set(title)
    dset = _token_set(description)

    senior_hit = len((tset | dset) & SENIOR_TOKENS) > 0
    phd_hit = len((tset | dset) & PHD_TOKENS) > 0

    # crude years signal
    years_hit = bool(re.search(r"\b([5-9]|10)\+?\s*(years|yrs)\b", (description or "").lower()))

    penalty = 0.0
    if senior_hit:
        penalty += 0.6
    if phd_hit:
        penalty += 0.6
    if years_hit:
        penalty += 0.6

    return min(1.0, penalty)

def score_skills(profile_skills_lc: List[str], job_text: str) -> Tuple[float, List[str]]:
    """
    Exact token substring check for skills. Returns (score in [0,1], matched skills).
    """
    jt = _norm(job_text)
    matched = []
    for s in profile_skills_lc:
        if not s:
            continue
        # allow skills like "c++", "power bi", "sql"
        if s in jt:
            matched.append(s)

    if not profile_skills_lc:
        return 0.0, []

    # normalize by skill list size, but cap
    raw = len(matched) / max(8, min(30, len(profile_skills_lc)))
    return min(1.0, raw), matched[:25]

def score_title(target_titles: List[str], job_title: str) -> Tuple[float, List[str]]:
    """
    Target title overlap using token overlap on each title phrase.
    """
    if not target_titles:
        return 0.0, []

    jtoks = _token_set(job_title)
    hits = []
    best = 0.0

    for tt in target_titles:
        ttoks = _token_set(tt)
        if not ttoks:
            continue
        overlap = len(jtoks & ttoks) / len(ttoks)
        if overlap > best:
            best = overlap
        if overlap >= 0.5:
            hits.append(tt)

    return min(1.0, best), hits[:10]

def score_resume_job_text(resume_tokens: set, job_text: str) -> Tuple[float, List[str]]:
    """
    Simple token overlap between resume and job (lightweight, deterministic).
    Returns score in [0,1] and a few evidence tokens.
    """
    jtoks = _token_set(job_text)
    if not resume_tokens or not jtoks:
        return 0.0, []

    overlap = resume_tokens & jtoks

    # downweight common junk
    stop = {"the", "and", "or", "with", "to", "in", "for", "of", "a", "an", "on"}
    overlap = {t for t in overlap if t not in stop and len(t) >= 3}

    score = min(1.0, len(overlap) / 120.0)
    top = sorted(list(overlap))[:25]
    return score, top

@dataclass
class RankedJob:
    score: float
    breakdown: Dict[str, float]
    matched_skills: List[str]
    title_hits: List[str]
    overlap_tokens: List[str]
    job: Dict[str, Any]

def rank_jobs(
    jobs: List[Dict[str, Any]],
    candidate_profile: Dict[str, Any],
    resume_text: str,
    weights: Dict[str, float] | None = None,
) -> List[RankedJob]:
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    sig = extract_profile_signals(candidate_profile, resume_text)
    prof_skills_lc = sig["skills_lc"]
    target_titles = sig["target_titles"]
    resume_tokens = sig["resume_tokens"]

    ranked: List[RankedJob] = []

    for j in jobs:
        title = j.get("title") or ""
        desc = j.get("description") or ""
        company = j.get("company") or ""
        location = j.get("location") or ""

        job_text = f"{title}\n{company}\n{location}\n{desc}"

        s_skills, matched = score_skills(prof_skills_lc, job_text)
        s_title, title_hits = score_title(target_titles, title)
        s_text, overlap_tokens = score_resume_job_text(resume_tokens, job_text)

        pen = seniority_penalty(title, desc)

        total = (
            w["skills"] * s_skills
            + w["title"] * s_title
            + w["resume_job_text"] * s_text
            - w["penalty"] * pen
        )

        total = max(0.0, min(1.0, total))

        ranked.append(
            RankedJob(
                score=total,
                breakdown={
                    "skills": float(s_skills),
                    "title": float(s_title),
                    "resume_job_text": float(s_text),
                    "penalty": float(pen),
                },
                matched_skills=matched,
                title_hits=title_hits,
                overlap_tokens=overlap_tokens,
                job=j,
            )
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked

def find_latest_resume_folder(resume_root: Path) -> Path | None:
    if not resume_root.exists():
        return None
    dirs = [p for p in resume_root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda p: p.name, reverse=True)[0]

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def load_jobs(path: Path) -> List[Dict[str, Any]]:
    """
    Loads jobs from:
      - List[dict] (expected)
      - {"jobs": [...]} or {"data": [...]} or {"items": [...]}
      - List[str] where each element is a JSON object string
      - NDJSON (one JSON object per line)
    Returns only dict rows.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return []

    # Try normal JSON first
    try:
        obj = json.loads(raw)
    except Exception:
        obj = None

    jobs: List[Any] = []

    if isinstance(obj, list):
        jobs = obj
    elif isinstance(obj, dict):
        for k in ("jobs", "data", "items", "results"):
            if isinstance(obj.get(k), list):
                jobs = obj[k]
                break
        if not jobs and "normalized" in obj and isinstance(obj["normalized"], list):
            jobs = obj["normalized"]
    else:
        # Fallback: NDJSON
        jobs = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                jobs.append(json.loads(line))
            except Exception:
                continue

    # If it's a list of JSON strings, decode each
    fixed: List[Dict[str, Any]] = []
    for row in jobs:
        if isinstance(row, dict):
            fixed.append(row)
            continue
        if isinstance(row, str):
            s = row.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    decoded = json.loads(s)
                    if isinstance(decoded, dict):
                        fixed.append(decoded)
                except Exception:
                    pass

    return fixed

