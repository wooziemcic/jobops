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
        max_tokens=2400,
    )

    return _safe_json_loads(repaired_text), repaired_text


def _missing_required_text_fields(data: Dict[str, Any]) -> list[str]:
    missing = []
    for k in REQUIRED_TEXT_FIELDS:
        v = data.get(k, "")
        if not isinstance(v, str) or not v.strip():
            missing.append(k)
    return missing


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
    """
    Second pass only for empty required fields. Returns updated data + raw text.
    """
    system = (
        "You are a strict JSON generator. "
        "Return ONLY valid JSON. No markdown. No extra keys. "
        "You must fill the requested fields with non-empty content."
    )

    user_obj = {
        "task": "Fill the missing fields in an application kit JSON.",
        "missing_fields_to_fill": missing_keys,
        "schema": expected_schema,
        "existing_json": base_data,
        "rules": [
            "Do not invent experience not present in resume/profile.",
            "Be specific and job-tailored.",
            "All requested fields must be NON-EMPTY strings.",
            "Return ONLY JSON object with the same keys as schema.",
        ],
        "context": {
            "candidate_profile": candidate_profile,
            "resume_text": (resume_text or "")[:5500],
            "job": {
                "title": job.get("title") or "",
                "company": job.get("company") or "",
                "location": job.get("location") or "",
                "apply_url": job.get("apply_url") or "",
                "description": (job.get("description") or "")[:2000],
            },
            "ranking_signals": ranking_signals,
        },
        "writing_guidance": {
            "recruiter_email_txt": "6–10 lines, professional, ask for quick chat, 1–2 resume proof points.",
            "linkedin_dm_txt": "2–4 lines, very short, friendly, direct.",
            "resume_tweak_suggestions_md": "Top 5 bullets + 3 before→after rewrites.",
            "match_report_md": "Concise but concrete; include evidence bullets.",
        },
    }

    text = _call_openai_legacy(
        api_key=api_key,
        model=model,
        system=system,
        user_obj=user_obj,
        temperature=0.0,
        max_tokens=2400,
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
    """
    Legacy OpenAI (openai==0.28.x) tailored application kit generator with:
      - strict JSON parsing
      - JSON repair if needed
      - required-field validation + regeneration if empty
    """
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

    ranking_signals = {
        "base_score": base_score,
        "breakdown": breakdown,
        "matched_skills": matched_skills,
        "title_hits": title_hits,
    }

    # Step A: generate full kit
    system = (
        "You are a strict JSON generator. "
        "Return ONLY valid JSON (double quotes, no trailing commas). "
        "You MUST include every key in the schema. "
        "All 4 main text fields must be non-empty strings: "
        "match_report_md, resume_tweak_suggestions_md, recruiter_email_txt, linkedin_dm_txt."
    )

    # Smaller, clearer instructions improve compliance
    user_obj = {
        "task": "Generate a job-tailored application kit.",
        "schema": expected_schema,
        "hard_requirements": [
            "Return ONLY JSON object",
            "Include every schema key",
            "No markdown fences",
            "The 4 main text fields must be NON-EMPTY strings",
        ],
        "candidate_profile": candidate_profile,
        "resume_text": resume_text[:5500],
        "job": {
            "title": job.get("title") or "",
            "company": job.get("company") or "",
            "location": job.get("location") or "",
            "apply_url": job.get("apply_url") or "",
            "description": job_desc[:2000],
        },
        "ranking_signals": ranking_signals,
        "style": {
            "one_line_pitch": "One sentence: value + proof + role.",
            "recruiter_email_txt": "6–10 lines, direct, 1–2 proof points, ask for quick chat.",
            "linkedin_dm_txt": "2–4 lines, concise.",
            "resume_tweak_suggestions_md": "Top 5 bullets + 3 before→after rewrites.",
        },
    }

    raw_model_text = _call_openai_legacy(
        api_key=api_key,
        model=model,
        system=system,
        user_obj=user_obj,
        temperature=float(temperature),
        max_tokens=2400,
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

    # Validate required fields; if empty, regenerate missing fields only
    missing = _missing_required_text_fields(data)
    if missing:
        try:
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
            # merge: take fixed values, but keep existing non-empty values
            for k in expected_schema.keys():
                if k in REQUIRED_TEXT_FIELDS:
                    if isinstance(fixed.get(k), str) and fixed.get(k).strip():
                        data[k] = fixed[k]
                else:
                    # for lists/other fields, prefer fixed if present
                    if fixed.get(k) is not None:
                        data[k] = fixed[k]
        except Exception:
            # If fill fails, we still return what we have (UI will show empties)
            pass

    # Normalize outputs
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
