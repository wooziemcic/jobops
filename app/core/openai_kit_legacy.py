# app/core/openai_kit_legacy.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional


def _safe_json_loads(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        # attempt to extract JSON object even if model added extra text
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start : end + 1])
        raise


def generate_application_kit_legacy(
    *,
    candidate_profile: Dict[str, Any],
    resume_text: str,
    job: Dict[str, Any],
    ranking_row: Optional[Dict[str, Any]] = None,
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Uses legacy OpenAI SDK (openai==0.28.x) to generate a tailored application kit.

    Returns dict with:
      - match_report_md
      - resume_tweak_suggestions_md
      - recruiter_email_txt
      - linkedin_dm_txt
      - keywords_to_emphasize (list)
      - must_have_hits (list)
      - gaps (list)
      - risk_flags (list)
      - one_line_pitch
    """
    try:
        import openai
    except Exception as e:
        raise RuntimeError("openai package not installed. Install openai==0.28.0") from e

    if api_key:
        openai.api_key = api_key

    # Keep payload bounded
    resume_text = (resume_text or "")[:6500]
    job_desc = (job.get("description") or "")[:2200]

    # Provide ranking signals if available (audit trail)
    base_score = None
    breakdown = None
    matched_skills = None
    title_hits = None
    if isinstance(ranking_row, dict):
        base_score = ranking_row.get("final_score") or ranking_row.get("score")
        breakdown = ranking_row.get("breakdown") or {}
        matched_skills = ranking_row.get("matched_skills") or []
        title_hits = ranking_row.get("title_hits") or []

    system = (
        "You are a strict JSON generator. "
        "Return ONLY valid JSON. No markdown fences, no extra commentary."
    )

    user_obj = {
        "task": "Generate a tailored application kit for a specific job using candidate resume/profile.",
        "output_contract": {
            "match_report_md": "string (markdown)",
            "resume_tweak_suggestions_md": "string (markdown)",
            "recruiter_email_txt": "string",
            "linkedin_dm_txt": "string",
            "keywords_to_emphasize": ["string"],
            "must_have_hits": ["string"],
            "gaps": ["string"],
            "risk_flags": ["string"],
            "one_line_pitch": "string",
        },
        "rules": [
            "Do not invent experience or credentials not in resume/profile.",
            "Be early-career appropriate; avoid overstating seniority.",
            "Make suggestions actionable and specific.",
            "Emails/DMs must be short, direct, and professional.",
            "Use the job’s keywords if they truly apply to the candidate.",
        ],
        "candidate_profile": candidate_profile,
        "resume_text": resume_text,
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
        "formatting": {
            "match_report_md_sections": [
                "Role summary (1–2 lines)",
                "Why you fit (bullet points)",
                "Evidence from resume (bullets with specific projects/tools)",
                "Gaps & mitigations (bullets)",
                "Interview angles (bullets: likely questions + your talking points)",
            ],
            "resume_tweak_suggestions_md_sections": [
                "Top 5 changes (bullets)",
                "Keyword alignment (bullets)",
                "Impact rewrites (before → after examples, 3 items)",
            ],
        },
    }

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj)},
        ],
        temperature=float(temperature),
        max_tokens=2200,
    )

    content = resp["choices"][0]["message"]["content"] if resp and resp.get("choices") else "{}"
    data = _safe_json_loads(content)

    # Normalize fields (avoid KeyError downstream)
    out = {
        "match_report_md": data.get("match_report_md", "").strip(),
        "resume_tweak_suggestions_md": data.get("resume_tweak_suggestions_md", "").strip(),
        "recruiter_email_txt": data.get("recruiter_email_txt", "").strip(),
        "linkedin_dm_txt": data.get("linkedin_dm_txt", "").strip(),
        "keywords_to_emphasize": data.get("keywords_to_emphasize", []) or [],
        "must_have_hits": data.get("must_have_hits", []) or [],
        "gaps": data.get("gaps", []) or [],
        "risk_flags": data.get("risk_flags", []) or [],
        "one_line_pitch": data.get("one_line_pitch", "").strip(),
        "raw": data,
    }
    return out
