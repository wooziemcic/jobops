# app/core/openai_kit_legacy.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def _extract_json_object(text: str) -> str:
    """Best-effort extraction of the outermost JSON object."""
    s = (text or "").strip()
    if not s:
        return s
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


def _light_json_sanitize(s: str) -> str:
    """
    Minimal sanitation for common model mistakes:
    - smart quotes -> normal quotes
    - remove trailing commas before } or ]
    """
    if not s:
        return s
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    # remove trailing commas like: {"a":1,} or [1,2,]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _safe_json_loads(s: str) -> Dict[str, Any]:
    raw = (s or "").strip()
    if not raw:
        return {}
    # 1) direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) extract object and sanitize
    extracted = _extract_json_object(raw)
    extracted = _light_json_sanitize(extracted)

    return json.loads(extracted)  # may still raise


def _repair_json_with_openai_legacy(
    *,
    bad_text: str,
    api_key: str,
    model: str,
    expected_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Second-pass: ask OpenAI to convert the previous output into STRICT valid JSON.
    """
    import openai  # openai==0.28.x

    openai.api_key = api_key

    system = (
        "You are a strict JSON formatter. "
        "You must output ONLY valid JSON with double quotes. "
        "No markdown. No commentary. No extra keys beyond the schema."
    )

    user_obj = {
        "task": "Repair invalid JSON into valid JSON that matches the schema exactly.",
        "schema": expected_schema,
        "invalid_output": bad_text,
        "requirements": [
            "Return ONLY a JSON object",
            "Use double quotes for all strings",
            "No trailing commas",
            "Ensure all required keys exist",
        ],
    }

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj)},
        ],
        temperature=0.0,
        max_tokens=2200,
    )

    content = resp["choices"][0]["message"]["content"] if resp and resp.get("choices") else "{}"
    return _safe_json_loads(content)


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
      - keywords_to_emphasize
      - must_have_hits
      - gaps
      - risk_flags
      - one_line_pitch
    """
    try:
        import openai
    except Exception as e:
        raise RuntimeError("openai package not installed. Install openai==0.28.0") from e

    if api_key:
        openai.api_key = api_key

    resume_text = (resume_text or "")[:6500]
    job_desc = (job.get("description") or "")[:2200]

    base_score = None
    breakdown = {}
    matched_skills = []
    title_hits = []
    if isinstance(ranking_row, dict):
        base_score = ranking_row.get("final_score") or ranking_row.get("score")
        breakdown = ranking_row.get("breakdown") or {}
        matched_skills = ranking_row.get("matched_skills") or []
        title_hits = ranking_row.get("title_hits") or []

    expected_schema = {
        "match_report_md": "string",
        "resume_tweak_suggestions_md": "string",
        "recruiter_email_txt": "string",
        "linkedin_dm_txt": "string",
        "keywords_to_emphasize": ["string"],
        "must_have_hits": ["string"],
        "gaps": ["string"],
        "risk_flags": ["string"],
        "one_line_pitch": "string",
    }

    system = (
        "You are a strict JSON generator. "
        "Return ONLY valid JSON. No markdown fences, no extra text."
    )

    user_obj = {
        "task": "Generate a tailored application kit for a specific job using candidate resume/profile.",
        "output_contract": expected_schema,
        "rules": [
            "Do not invent experience or credentials not in resume/profile.",
            "Be early-career appropriate; avoid overstating seniority.",
            "Make suggestions actionable and specific.",
            "Emails/DMs must be short, direct, and professional.",
            "Use the job’s keywords only if they truly apply to the candidate.",
            "Return ONLY JSON. No backticks. No markdown.",
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

    # Parse with repair fallback
    try:
        data = _safe_json_loads(content)
    except Exception:
        if not api_key:
            raise
        data = _repair_json_with_openai_legacy(
            bad_text=content,
            api_key=api_key,
            model=model,
            expected_schema=expected_schema,
        )

    # Normalize outputs (avoid downstream KeyError)
    out = {
        "match_report_md": (data.get("match_report_md", "") or "").strip(),
        "resume_tweak_suggestions_md": (data.get("resume_tweak_suggestions_md", "") or "").strip(),
        "recruiter_email_txt": (data.get("recruiter_email_txt", "") or "").strip(),
        "linkedin_dm_txt": (data.get("linkedin_dm_txt", "") or "").strip(),
        "keywords_to_emphasize": data.get("keywords_to_emphasize", []) or [],
        "must_have_hits": data.get("must_have_hits", []) or [],
        "gaps": data.get("gaps", []) or [],
        "risk_flags": data.get("risk_flags", []) or [],
        "one_line_pitch": (data.get("one_line_pitch", "") or "").strip(),
        "raw": data,
    }
    return out
