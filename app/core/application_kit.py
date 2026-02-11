# app/core/application_kit.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TOKEN_RE = re.compile(r"[a-zA-Z0-9+#\.]+")

COMMON_SKILL_ALIASES = {
    "powerbi": "Power BI",
    "power bi": "Power BI",
    "tableau": "Tableau",
    "sql": "SQL",
    "python": "Python",
    "excel": "Excel",
    "alteryx": "Alteryx",
    "snowflake": "Snowflake",
    "databricks": "Databricks",
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "GCP",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "scikit": "scikit-learn",
    "scikit-learn": "scikit-learn",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "nlp": "NLP",
}

STOPWORDS = {
    "the","and","or","with","to","in","for","of","a","an","on","as","is","are","be","you","we","our","their"
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

def extract_profile_basics(profile: Dict[str, Any]) -> Dict[str, Any]:
    # Keep schema-agnostic.
    name = (
        _safe_get(profile, ["name"], None)
        or _safe_get(profile, ["candidate", "name"], None)
        or _safe_get(profile, ["profile", "name"], None)
        or ""
    )
    email = (
        _safe_get(profile, ["email"], None)
        or _safe_get(profile, ["candidate", "email"], None)
        or _safe_get(profile, ["profile", "email"], None)
        or ""
    )
    phone = (
        _safe_get(profile, ["phone"], None)
        or _safe_get(profile, ["candidate", "phone"], None)
        or _safe_get(profile, ["profile", "phone"], None)
        or ""
    )
    links = (
        _safe_get(profile, ["links"], None)
        or _safe_get(profile, ["candidate", "links"], None)
        or _safe_get(profile, ["profile", "links"], None)
        or []
    )
    if isinstance(links, str):
        links = [links]
    if not isinstance(links, list):
        links = []

    skills = (
        _safe_get(profile, ["skills"], [])
        or _safe_get(profile, ["candidate", "skills"], [])
        or _safe_get(profile, ["profile", "skills"], [])
        or []
    )
    if isinstance(skills, dict):
        flat = []
        for v in skills.values():
            if isinstance(v, list):
                flat.extend(v)
        skills = flat
    skills = sorted({str(s).strip() for s in skills if str(s).strip()})

    target_titles = (
        _safe_get(profile, ["target_titles"], [])
        or _safe_get(profile, ["candidate", "target_titles"], [])
        or _safe_get(profile, ["profile", "target_titles"], [])
        or []
    )
    if isinstance(target_titles, str):
        target_titles = [target_titles]
    if not isinstance(target_titles, list):
        target_titles = []
    target_titles = [str(t).strip() for t in target_titles if str(t).strip()]

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "links": links,
        "skills": skills,
        "target_titles": target_titles,
    }

def canonical_skill_display(skill: str) -> str:
    s = _norm(skill)
    return COMMON_SKILL_ALIASES.get(s, skill)

def extract_job_basics(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": job.get("title") or "",
        "company": job.get("company") or "",
        "location": job.get("location") or "",
        "apply_url": job.get("apply_url") or "",
        "source": job.get("source") or "",
        "posted_at": job.get("posted_at") or job.get("date") or "",
        "description": (job.get("description") or "").strip(),
    }

def infer_job_skills_from_text(job_text: str, profile_skills: List[str]) -> Tuple[List[str], List[str]]:
    """
    Returns (matched_skills, missing_skills_suggestions)
    - matched: profile skills that appear in job text
    - missing suggestions: high-signal skills present in job text but not in profile skills
      (limited heuristic via aliases + common tokens)
    """
    jt = _norm(job_text)
    prof_norm = {_norm(s) for s in profile_skills}

    matched = []
    for s in profile_skills:
        if _norm(s) and _norm(s) in jt:
            matched.append(canonical_skill_display(s))

    # Missing suggestions: scan aliases keys in job text that aren't in profile
    suggested = []
    for raw, disp in COMMON_SKILL_ALIASES.items():
        if raw in jt and raw not in prof_norm:
            suggested.append(disp)

    # de-dupe preserve order
    def uniq(xs):
        out = []
        seen = set()
        for x in xs:
            k = _norm(x)
            if k and k not in seen:
                out.append(x)
                seen.add(k)
        return out

    return uniq(matched)[:25], uniq(suggested)[:20]

def extract_talking_points(resume_text: str, job_desc: str, max_points: int = 6) -> List[str]:
    """
    Deterministic talking points: token overlap terms -> short bullets.
    Not "fluent", but audit-friendly and useful.
    """
    rset = _token_set(resume_text)
    jset = _token_set(job_desc)

    overlap = {t for t in (rset & jset) if t not in STOPWORDS and len(t) >= 4}
    top = sorted(list(overlap))[:40]

    points = []
    # Convert tokens into thematic bullets
    buckets = {
        "Analytics & BI": {"sql","tableau","power","bi","excel","dashboard","reporting","kpi"},
        "Data Engineering": {"etl","pipeline","snowflake","databricks","spark","warehouse","lake"},
        "ML/AI": {"model","training","inference","ml","ai","classification","regression","nlp"},
        "Cloud": {"aws","azure","gcp","s3","lambda","ec2"},
        "Ops & Delivery": {"agile","scrum","stakeholder","requirements","roadmap","delivery"},
    }

    used = set()
    for theme, toks in buckets.items():
        if any(t in overlap for t in toks):
            hit_words = sorted(list(set(toks) & overlap))[:6]
            if hit_words:
                points.append(f"{theme}: emphasize experience with {', '.join(hit_words)}.")
                used |= set(hit_words)

    # Fill remainder with strongest leftover tokens
    leftover = [t for t in top if t not in used][:max(0, max_points - len(points))]
    for t in leftover:
        points.append(f"Use concrete example related to: {t}.")
        if len(points) >= max_points:
            break

    return points[:max_points]

def resume_tweak_suggestions(profile_skills: List[str], matched_skills: List[str], suggested_missing: List[str]) -> List[str]:
    sugg = []

    if suggested_missing:
        sugg.append(
            "Add a short 'Tools/Skills' line near the top that explicitly includes: "
            + ", ".join(suggested_missing[:10])
            + ". (Only include what you truly have exposure to.)"
        )

    if matched_skills:
        sugg.append(
            "Mirror the job language by weaving these matched skills into 1–2 bullets in your most relevant experience: "
            + ", ".join(matched_skills[:10])
            + "."
        )

    # Generic deterministic suggestions
    sugg.extend(
        [
            "Add a 2–3 line summary tailored to this role: impact + tools + domain.",
            "Ensure your strongest quantified outcomes (%, $, time saved) appear in the first half of the resume.",
            "If the job mentions dashboards/reporting, include one bullet describing a dashboard you built: audience, KPIs, tooling.",
        ]
    )
    return sugg[:8]

def build_recruiter_email(
    candidate_name: str,
    candidate_email: str,
    company: str,
    title: str,
    apply_url: str,
    matched_skills: List[str],
    talking_points: List[str],
) -> str:
    skills_line = ", ".join(matched_skills[:6]) if matched_skills else "relevant analytics and delivery experience"
    url_line = f"\nApply link: {apply_url}\n" if apply_url else "\n"

    bullets = ""
    for p in talking_points[:3]:
        bullets += f"- {p}\n"
    if not bullets:
        bullets = "- Strong fit based on skills and past project impact.\n"

    return (
        f"Subject: Interest in {title} at {company}\n\n"
        f"Hi {company} Recruiting Team,\n\n"
        f"I'm reaching out regarding the {title} role. My background aligns well with what you're looking for, "
        f"especially around {skills_line}.\n\n"
        f"Relevant highlights:\n{bullets}\n"
        f"I’d appreciate the opportunity to share a quick overview of my experience and learn more about the team’s priorities.\n"
        f"{url_line}"
        f"Best regards,\n"
        f"{candidate_name or '[Your Name]'}\n"
        f"{candidate_email or '[your.email@example.com]'}\n"
    )

def build_linkedin_dm(
    candidate_name: str,
    company: str,
    title: str,
    matched_skills: List[str],
) -> str:
    skills_line = ", ".join(matched_skills[:5]) if matched_skills else "analytics and delivery"
    return (
        f"Hi — I’m interested in the {title} role at {company}. My background includes {skills_line}. "
        f"If you’re open to it, I’d love to ask 1–2 quick questions about the team and what strong candidates do well. "
        f"Thanks in advance.\n\n"
        f"- {candidate_name or '[Your Name]'}"
    )

def render_match_report_md(
    job: Dict[str, Any],
    profile: Dict[str, Any],
    resume_text: str,
    matched_skills: List[str],
    missing_suggestions: List[str],
    talking_points: List[str],
) -> str:
    jb = extract_job_basics(job)
    pb = extract_profile_basics(profile)

    links_line = ""
    if pb["links"]:
        links_line = "\n".join([f"- {x}" for x in pb["links"][:8]])

    job_desc = jb["description"] or ""
    preview = job_desc[:1200] + ("..." if len(job_desc) > 1200 else "")

    return (
        f"# Match Report\n\n"
        f"## Job\n"
        f"- Company: {jb['company']}\n"
        f"- Title: {jb['title']}\n"
        f"- Location: {jb['location']}\n"
        f"- Source: {jb['source']}\n"
        f"- Posted: {jb['posted_at']}\n"
        f"- Apply: {jb['apply_url']}\n\n"
        f"## Candidate\n"
        f"- Name: {pb['name']}\n"
        f"- Email: {pb['email']}\n"
        f"- Phone: {pb['phone']}\n\n"
        f"### Links\n"
        f"{links_line or '- (none)'}\n\n"
        f"## Skills alignment\n"
        f"### Matched skills found in job text\n"
        f"{('- ' + '\\n- '.join(matched_skills[:20])) if matched_skills else '- (none detected)'}\n\n"
        f"### Skills suggested by job text (missing from profile)\n"
        f"{('- ' + '\\n- '.join(missing_suggestions[:20])) if missing_suggestions else '- (none detected)'}\n\n"
        f"## Talking points for interviews / screening\n"
        f"{('- ' + '\\n- '.join(talking_points)) if talking_points else '- (none)'}\n\n"
        f"## Job description preview\n"
        f"{preview}\n"
    )

def save_kit(
    kit_root: Path,
    job: Dict[str, Any],
    profile: Dict[str, Any],
    resume_text: str,
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Stable-ish folder name
    company = re.sub(r"[^a-zA-Z0-9]+", "_", (job.get("company") or "Unknown"))[:40].strip("_")
    title = re.sub(r"[^a-zA-Z0-9]+", "_", (job.get("title") or "Role"))[:50].strip("_")
    kit_dir = kit_root / f"{ts}__{company}__{title}"
    kit_dir.mkdir(parents=True, exist_ok=True)

    jb = extract_job_basics(job)
    pb = extract_profile_basics(profile)

    job_text = f"{jb['title']}\n{jb['company']}\n{jb['location']}\n{jb['description']}"

    matched, missing = infer_job_skills_from_text(job_text, pb["skills"])
    talking = extract_talking_points(resume_text, jb["description"], max_points=6)
    tweaks = resume_tweak_suggestions(pb["skills"], matched, missing)

    match_md = render_match_report_md(job, profile, resume_text, matched, missing, talking)
    recruiter_email = build_recruiter_email(pb["name"], pb["email"], jb["company"], jb["title"], jb["apply_url"], matched, talking)
    linkedin_dm = build_linkedin_dm(pb["name"], jb["company"], jb["title"], matched)

    (kit_dir / "match_report.md").write_text(match_md, encoding="utf-8")
    (kit_dir / "recruiter_email.txt").write_text(recruiter_email, encoding="utf-8")
    (kit_dir / "linkedin_dm.txt").write_text(linkedin_dm, encoding="utf-8")
    (kit_dir / "resume_tweak_suggestions.md").write_text(
        "# Resume tweak suggestions\n\n" + "\n".join([f"- {x}" for x in tweaks]) + "\n",
        encoding="utf-8",
    )

    (kit_dir / "job_snapshot.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    (kit_dir / "profile_snapshot.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

    # Tiny manifest for traceability
    manifest = {
        "created_at_utc": ts,
        "job": {"company": jb["company"], "title": jb["title"], "apply_url": jb["apply_url"]},
        "candidate": {"name": pb["name"], "email": pb["email"]},
        "artifacts": [
            "match_report.md",
            "recruiter_email.txt",
            "linkedin_dm.txt",
            "resume_tweak_suggestions.md",
            "job_snapshot.json",
            "profile_snapshot.json",
        ],
    }
    (kit_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return kit_dir
