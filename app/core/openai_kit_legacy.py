# app/core/openai_kit_legacy.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional


def _call_openai_legacy_text(
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> str:
    import openai  # openai==0.28.x

    openai.api_key = api_key
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    return resp["choices"][0]["message"]["content"] if resp and resp.get("choices") else ""


def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s[:n] if len(s) > n else s


def generate_application_kit_legacy(
    *,
    candidate_profile: Dict[str, Any],
    resume_text: str,
    job: Dict[str, Any],
    ranking_row: Optional[Dict[str, Any]] = None,
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    Text-first kit generation (robust on Streamlit Cloud).
    Avoids JSON parsing entirely.

    Returns:
      match_report_md, resume_tweak_suggestions_md, recruiter_email_txt, linkedin_dm_txt
      plus debug raw strings.
    """
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI kit generation.")

    resume_text = _truncate(resume_text or "", 6500)
    job_desc = _truncate(job.get("description") or "", 2400)

    # ranking signals (optional)
    base_score = None
    breakdown = {}
    matched_skills = []
    title_hits = []
    if isinstance(ranking_row, dict):
        base_score = ranking_row.get("final_score") or ranking_row.get("score")
        breakdown = ranking_row.get("breakdown") or {}
        matched_skills = ranking_row.get("matched_skills") or []
        title_hits = ranking_row.get("title_hits") or []

    context = {
        "job": {
            "title": job.get("title") or "",
            "company": job.get("company") or "",
            "location": job.get("location") or "",
            "apply_url": job.get("apply_url") or "",
            "source": job.get("source") or "",
            "posted_at": job.get("posted_at") or job.get("date") or "",
            "description": job_desc,
        },
        "ranking_signals": {
            "base_score": base_score,
            "breakdown": breakdown,
            "matched_skills": matched_skills,
            "title_hits": title_hits,
        },
        "candidate_profile": candidate_profile,
        "resume_text": resume_text,
    }

    # Common system: forbid placeholders/guidelines
    system_common = (
        "You write final, job-tailored application materials. "
        "Do NOT write instructions, templates, or placeholders. "
        "Write the actual final text. "
        "Do not invent experience; only use what is present in the resume/profile. "
        "Be specific, include concrete proof points (tools, projects, metrics) when available."
    )

    ctx_block = json.dumps(context, indent=2)

    # 1) Match report (markdown)
    system = system_common + " Output MUST be in Markdown."
    user = (
        "Create a MATCH REPORT in Markdown with these sections:\n"
        "## Role summary\n"
        "## Why I fit (bullets)\n"
        "## Evidence from resume (bullets with tools/projects/metrics)\n"
        "## Gaps & mitigations\n"
        "## Interview angles (5 bullets)\n\n"
        "Use the context JSON below.\n\n"
        f"{ctx_block}\n"
    )
    raw_match = _call_openai_legacy_text(
        api_key=api_key,
        model=model,
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=1400,
    ).strip()

    # 2) Resume tweaks (markdown)
    system = system_common + " Output MUST be in Markdown."
    user = (
        "Create RESUME TWEAK SUGGESTIONS in Markdown with:\n"
        "## Top 5 changes (bullets)\n"
        "## Keyword alignment (bullets)\n"
        "## Impact rewrites (before → after) x3\n"
        "Each rewrite must reference a real resume item and make it more job-relevant.\n\n"
        "Use the context JSON below.\n\n"
        f"{ctx_block}\n"
    )
    raw_tweaks = _call_openai_legacy_text(
        api_key=api_key,
        model=model,
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=1400,
    ).strip()

    # 3) Recruiter email (plain text)
    system = system_common + " Output MUST be plain text."
    user = (
        "Write a recruiter email (8–12 lines) tailored to this role/company.\n"
        "Requirements:\n"
        "- Direct subject line on first line like: Subject: ...\n"
        "- 1–2 proof points from the resume (tools/projects/metrics)\n"
        "- Mention why this role is a fit\n"
        "- End with a clear ask for a quick 15-min chat\n"
        "- No placeholders\n\n"
        "Use the context JSON below.\n\n"
        f"{ctx_block}\n"
    )
    raw_email = _call_openai_legacy_text(
        api_key=api_key,
        model=model,
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=700,
    ).strip()

    # 4) LinkedIn DM (plain text)
    system = system_common + " Output MUST be plain text."
    user = (
        "Write a LinkedIn DM (3–5 lines) tailored to this role/company.\n"
        "Requirements:\n"
        "- concise\n"
        "- 1 proof point from the resume\n"
        "- ask if they'd be open to a quick chat\n"
        "- no placeholders\n\n"
        "Use the context JSON below.\n\n"
        f"{ctx_block}\n"
    )
    raw_dm = _call_openai_legacy_text(
        api_key=api_key,
        model=model,
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=350,
    ).strip()

    # Simple sanity fallback if model returns nothing
    def _fallback(name: str) -> str:
        return f"(OpenAI returned empty output for {name}. Check OPENAI_API_KEY/model access.)"

    match_report_md = raw_match if len(raw_match) > 40 else _fallback("match_report_md")
    resume_tweak_suggestions_md = raw_tweaks if len(raw_tweaks) > 40 else _fallback("resume_tweak_suggestions_md")
    recruiter_email_txt = raw_email if len(raw_email) > 40 else _fallback("recruiter_email_txt")
    linkedin_dm_txt = raw_dm if len(raw_dm) > 20 else _fallback("linkedin_dm_txt")

    # Keep old keys so your manifest/UI doesn't break
    return {
        "match_report_md": match_report_md,
        "resume_tweak_suggestions_md": resume_tweak_suggestions_md,
        "recruiter_email_txt": recruiter_email_txt,
        "linkedin_dm_txt": linkedin_dm_txt,
        "keywords_to_emphasize": [],
        "must_have_hits": [],
        "gaps": [],
        "risk_flags": [],
        "one_line_pitch": "",
        # debug
        "raw_model_text": "",  # not used in text-first mode
        "raw_repair_text": "",
        "raw_fill_text": "",
        "raw": {
            "match_report_md": raw_match,
            "resume_tweak_suggestions_md": raw_tweaks,
            "recruiter_email_txt": raw_email,
            "linkedin_dm_txt": raw_dm,
        },
    }
