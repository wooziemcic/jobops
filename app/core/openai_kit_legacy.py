# app/core/openai_kit_legacy.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple


REQUIRED_TEXT_FIELDS = [
    "match_report_md",
    "resume_tweak_suggestions_md",
    "recruiter_email_txt",
    "linkedin_dm_txt",
]

# Heuristics to detect "the model repeated instructions instead of generating content"
INSTRUCTION_PATTERNS = [
    r"\binclude top\b",
    r"\btop\s*5\b",
    r"\bbefore[-\s]*after\b",
    r"\b6[-\s]*10 lines\b",
    r"\b2[-\s]*4 lines\b",
    r"\bkeep it concise\b",
    r"\bask for a quick chat\b",
    r"\bfor maximum impact\b",
    r"\buse the job\b",
    r"\bmust be\b",
    r"\byou should\b",
    r"\bwrite\b\s+\d",
    r"\breturn only\b",
]

# Minimum lengths so we don't accept "one sentence"
MIN_LEN = {
    "match_report_md": 400,
    "resume_tweak_suggestions_md": 400,
    "recruiter_email_txt": 250,
    "linkedin_dm_txt": 120,
}


def _extract_json_object(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


def _light_json_sanitize(s: str) -> str:
    if not s:
        return s
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
    return s


def _safe_json_loads(s: str) -> Dict[str, Any]:
    raw = (s or "").strip()
    if not raw:
        return {}

    try:
        return json.loads(raw)
    except Exception:
        pass

    extracted = _extract_json_object(raw)
    extracted = _light_json_sanitize(extracted)
    return json.loads(extracted)


def _call_openai_legacy(
    *,
    api_key: str,
    model: str,
    system: str,
    user_obj: Dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> str:
    import openai  # openai==0.28.x

    openai.api_key = api_key

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj)},
        ],
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    return resp["choices"][0]["message"]["content"] if resp and resp.get("choices") else ""


def _repair_json_with_openai_legacy(
    *,
    bad_text: str,
    api_key: str,
    model: str,
    expected_schema: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    system = (
        "You are a strict JSON formatter. "
        "Output ONLY valid JSON with double quotes. "
        "No markdown. No commentary. "
        "You MUST include every key in the schema."
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
            "If a field is unknown, return an empty string or empty list (as appropriate)",
        ],
    }

    repaired_text = _call_openai_legacy(
        api_key=api_key,
        model=model,
        system=system,
        user_obj=user_obj,
        temperature=0.0,
        max_tokens=2600,
    )

    return _safe_json_loads(repaired_text), repaired_text


def _looks_like_instructions(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    for pat in INSTRUCTION_PATTERNS:
        if re.search(pat, t):
            return True
    return False


def _missing_or_bad_fields(data: Dict[str, Any], expected_keys: list[str]) -> list[str]:
    """
    Return list of fields that are missing OR too short OR look like instructions.
    Only applies strict checks to the 4 main text fields.
    """
    missing = []

    # Ensure all expected keys exist
    for k in expected_keys:
        if k not in data:
            missing.append(k)

    # Validate required text fields
    for k in REQUIRED_TEXT_FIELDS:
        v = data.get(k, "")
        if not isinstance(v, str) or not v.strip():
            missing.append(k)
            continue
        if _looks_like_instructions(v):
            missing.append(k)
            continue
        if len(v.strip()) < MIN_LEN.get(k, 0):
            missing.append(k)
            continue

    # Deduplicate while preserving order
    seen = set()
    out = []
    for k in missing:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _fill_missing_fields(
    *,
    api_key: str,
    model: str,
    expected_schema: Dict[str, Any],
    base_data: Dict[str, Any],
    missing_keys: list[str],
    candidate_profile: Dict[str, Any],
    resume_text: str,
    job: Dict[str, Any],
    ranking_signals: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    system = (
        "You are a strict JSON generator. "
        "Return ONLY valid JSON. No markdown. No extra keys. "
        "You must fill the requested fields with REAL content (not instructions). "
        "Do NOT describe what you will do; DO the writing."
    )

    user_obj = {
        "task": "Fill missing/invalid fields in an application kit JSON with real final text.",
        "missing_fields_to_fill": missing_keys,
        "schema": expected_schema,
        "existing_json": base_data,
        "rules": [
            "Do not invent experience not present in resume/profile.",
            "Be specific and job-tailored.",
            "For the 4 main fields, produce real final copy, not guidelines.",
            "Use concrete proof points from resume/profile.",
            "Return ONLY JSON object with exactly the schema keys.",
        ],
        "hard_minimum_lengths": MIN_LEN,
        "context": {
            "candidate_profile": candidate_profile,
            "resume_text": (resume_text or "")[:6000],
            "job": {
                "title": job.get("title") or "",
                "company": job.get("company") or "",
                "location": job.get("location") or "",
                "apply_url": job.get("apply_url") or "",
                "description": (job.get("description") or "")[:2200],
            },
            "ranking_signals": ranking_signals,
        },
        "output_formatting": {
            "match_report_md": {
                "must_include_sections": [
                    "Role summary",
                    "Why you fit",
                    "Evidence from resume",
                    "Gaps & mitigations",
                    "Interview angles",
                ],
                "bullet_style": "Use bullets where appropriate, include tools/projects/metrics.",
            },
            "resume_tweak_suggestions_md": {
                "must_include": [
                    "Top 5 changes",
                    "Keyword alignment",
                    "Impact rewrites (before → after) x3",
                ]
            },
            "recruiter_email_txt": "Write the final email (no placeholders), 8–12 lines, specific to role/company.",
            "linkedin_dm_txt": "Write the final DM (no placeholders), 3–5 lines, specific to role/company.",
        },
    }

    text = _call_openai_legacy(
        api_key=api_key,
        model=model,
        system=system,
        user_obj=user_obj,
        temperature=0.0,
        max_tokens=2600,
    )

    fixed = _safe_json_loads(text)
    return fixed, text


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
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI kit generation.")

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
    expected_keys = list(expected_schema.keys())

    ranking_signals = {
        "base_score": base_score,
        "breakdown": breakdown,
        "matched_skills": matched_skills,
        "title_hits": title_hits,
    }

    system = (
        "You are a strict JSON generator. "
        "Return ONLY valid JSON (double quotes, no trailing commas). "
        "You MUST include every key in the schema. "
        "For the 4 main text fields, write REAL final content (not instructions)."
    )

    user_obj = {
        "task": "Generate a job-tailored application kit with real final text.",
        "schema": expected_schema,
        "hard_requirements": [
            "Return ONLY JSON object",
            "Include every schema key",
            "No markdown fences",
            "No placeholders or guidelines text",
            "Main fields must be real content with minimum length constraints",
        ],
        "hard_minimum_lengths": MIN_LEN,
        "candidate_profile": candidate_profile,
        "resume_text": resume_text[:6000],
        "job": {
            "title": job.get("title") or "",
            "company": job.get("company") or "",
            "location": job.get("location") or "",
            "apply_url": job.get("apply_url") or "",
            "description": job_desc[:2200],
        },
        "ranking_signals": ranking_signals,
    }

    raw_model_text = _call_openai_legacy(
        api_key=api_key,
        model=model,
        system=system,
        user_obj=user_obj,
        temperature=float(temperature),
        max_tokens=2600,
    )

    raw_repair_text = ""
    raw_fill_text = ""

    # Parse or repair JSON
    try:
        data = _safe_json_loads(raw_model_text)
    except Exception:
        data, raw_repair_text = _repair_json_with_openai_legacy(
            bad_text=raw_model_text,
            api_key=api_key,
            model=model,
            expected_schema=expected_schema,
        )

    # Fill missing/bad fields (too short / instruction-like / missing keys)
    missing = _missing_or_bad_fields(data, expected_keys)
    if missing:
        fixed, raw_fill_text = _fill_missing_fields(
            api_key=api_key,
            model=model,
            expected_schema=expected_schema,
            base_data=data,
            missing_keys=missing,
            candidate_profile=candidate_profile,
            resume_text=resume_text,
            job=job,
            ranking_signals=ranking_signals,
        )

        # Merge fixed into data for the keys we asked to fill
        for k in expected_keys:
            if k in fixed:
                data[k] = fixed[k]

    # Final normalization
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
        "raw_model_text": raw_model_text or "",
        "raw_repair_text": raw_repair_text or "",
        "raw_fill_text": raw_fill_text or "",
        "raw": data,
    }
    return out
