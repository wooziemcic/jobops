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

INSTRUCTION_PATTERNS = [
    r"\binclude top\b",
    r"\btop\s*5\b",
    r"\bbefore[-\s]*after\b",
    r"\b6[-\s]*10 lines\b",
    r"\b2[-\s]*4 lines\b",
    r"\bkeep it concise\b",
    r"\bask for a quick chat\b",
    r"\bfor maximum impact\b",
    r"\breturn only\b",
    r"\bno markdown\b",
]

MIN_LEN = {
    "match_report_md": 300,
    "resume_tweak_suggestions_md": 300,
    "recruiter_email_txt": 180,
    "linkedin_dm_txt": 80,
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
    # remove weird null bytes
    s = s.replace("\x00", "")
    return s


def _try_json_loads(s: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns (obj, error_str). obj is None if parsing fails.
    """
    try:
        return json.loads(s), None
    except Exception as e:
        return None, str(e)


def _safe_json_loads_no_throw(s: str) -> Tuple[Dict[str, Any], str]:
    """
    Never throws. Returns (parsed_obj_or_empty, debug_reason).
    """
    raw = (s or "").strip()
    if not raw:
        return {}, "empty_input"

    # 1) direct
    obj, err = _try_json_loads(raw)
    if obj is not None and isinstance(obj, dict):
        return obj, "direct_ok"

    # 2) extract braces
    extracted = _extract_json_object(raw)
    extracted = _light_json_sanitize(extracted)
    obj, err2 = _try_json_loads(extracted)
    if obj is not None and isinstance(obj, dict):
        return obj, "extracted_ok"

    # 3) final sanitize (sometimes model returns JSON with leading/trailing text)
    extracted2 = re.sub(r"^[^{]*", "", raw)
    extracted2 = re.sub(r"[^}]*$", "", extracted2)
    extracted2 = _light_json_sanitize(extracted2)
    obj, err3 = _try_json_loads(extracted2)
    if obj is not None and isinstance(obj, dict):
        return obj, "trimmed_ok"

    return {}, f"parse_failed: direct={err} extracted={err2} trimmed={err3}"


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


def _looks_like_instructions(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    for pat in INSTRUCTION_PATTERNS:
        if re.search(pat, t):
            return True
    return False


def _missing_or_bad_fields(data: Dict[str, Any], expected_keys: list[str]) -> list[str]:
    missing = []

    # Ensure all expected keys exist
    for k in expected_keys:
        if k not in data:
            missing.append(k)

    # Validate main text fields
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

    # Deduplicate keep order
    seen = set()
    out = []
    for k in missing:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _strict_json_only_pass(
    *,
    api_key: str,
    model: str,
    expected_schema: Dict[str, Any],
    text_to_convert: str,
) -> Tuple[Dict[str, Any], str, str]:
    """
    Final fallback: ask the model to output STRICT JSON only from arbitrary text.
    Never throws. Returns (parsed, raw_text, parse_debug).
    """
    system = (
        "You are a strict JSON formatter. "
        "Return ONLY a JSON object. "
        "No markdown. No commentary. "
        "Use double quotes. No trailing commas. "
        "Include every key in the schema."
    )
    user_obj = {
        "task": "Convert the provided text into valid JSON that matches the schema exactly.",
        "schema": expected_schema,
        "text": text_to_convert,
    }

    raw = _call_openai_legacy(
        api_key=api_key,
        model=model,
        system=system,
        user_obj=user_obj,
        temperature=0.0,
        max_tokens=2600,
    )
    parsed, dbg = _safe_json_loads_no_throw(raw)
    return parsed, raw, dbg


def _repair_json_with_openai_legacy(
    *,
    bad_text: str,
    api_key: str,
    model: str,
    expected_schema: Dict[str, Any],
) -> Tuple[Dict[str, Any], str, str]:
    """
    Tries to repair bad JSON. Never throws.
    Returns (parsed_or_empty, raw_repair_text, parse_debug_reason)
    """
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

    parsed, dbg = _safe_json_loads_no_throw(repaired_text)
    return parsed, repaired_text, dbg


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
) -> Tuple[Dict[str, Any], str, str]:
    system = (
        "You are a strict JSON generator. "
        "Return ONLY valid JSON. No markdown. No extra keys. "
        "Write REAL final content, not guidelines. "
        "Do NOT describe what you will do; DO the writing."
    )

    user_obj = {
        "task": "Fill missing/invalid fields with real final text.",
        "missing_fields_to_fill": missing_keys,
        "schema": expected_schema,
        "existing_json": base_data,
        "hard_minimum_lengths": MIN_LEN,
        "rules": [
            "Do not invent experience not present in resume/profile.",
            "Be specific and job-tailored.",
            "Main text fields must be real copy, not placeholders.",
            "Return ONLY JSON with exactly the schema keys.",
        ],
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
    }

    text = _call_openai_legacy(
        api_key=api_key,
        model=model,
        system=system,
        user_obj=user_obj,
        temperature=0.0,
        max_tokens=2600,
    )

    parsed, dbg = _safe_json_loads_no_throw(text)
    return parsed, text, dbg


def _ensure_schema_defaults(expected_keys: list[str], data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make sure all expected keys exist with safe defaults.
    """
    out = dict(data) if isinstance(data, dict) else {}
    for k in expected_keys:
        if k not in out:
            # default by type guess
            if k in ("keywords_to_emphasize", "must_have_hits", "gaps", "risk_flags"):
                out[k] = []
            else:
                out[k] = ""
    return out


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

    # Step A: initial generation
    system = (
        "You are a strict JSON generator. Return ONLY valid JSON. "
        "Include every schema key. No markdown. "
        "Write REAL final content, not guidelines. "
        "Main text fields must be substantive."
    )

    user_obj = {
        "task": "Generate a job-tailored application kit with real final text.",
        "schema": expected_schema,
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
    raw_strict_text = ""
    parse_debug = "none"
    repair_parse_debug = ""
    fill_parse_debug = ""
    strict_parse_debug = ""

    data, parse_debug = _safe_json_loads_no_throw(raw_model_text)

    # If initial parse fails or returns empty, attempt repair
    if not data:
        data, raw_repair_text, repair_parse_debug = _repair_json_with_openai_legacy(
            bad_text=raw_model_text,
            api_key=api_key,
            model=model,
            expected_schema=expected_schema,
        )

        # If repair still fails, strict-json-only pass
        if not data:
            data, raw_strict_text, strict_parse_debug = _strict_json_only_pass(
                api_key=api_key,
                model=model,
                expected_schema=expected_schema,
                text_to_convert=raw_repair_text or raw_model_text,
            )

    # Ensure schema defaults so downstream doesn't KeyError
    data = _ensure_schema_defaults(expected_keys, data)

    # Fill missing/bad fields (too short / placeholder / missing keys)
    missing = _missing_or_bad_fields(data, expected_keys)
    if missing:
        fixed, raw_fill_text, fill_parse_debug = _fill_missing_fields(
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

        if fixed:
            fixed = _ensure_schema_defaults(expected_keys, fixed)
            for k in expected_keys:
                data[k] = fixed.get(k, data.get(k))

    # Final guarantee: never crash; if still bad, provide safe stubs
    for k in REQUIRED_TEXT_FIELDS:
        v = data.get(k, "")
        if not isinstance(v, str) or not v.strip() or _looks_like_instructions(v):
            data[k] = f"(Generation failed for {k}. See openai_raw_output.txt / openai_fill_output.txt.)"

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
        # extra debug (optional)
        "raw_strict_text": raw_strict_text or "",
        "parse_debug": parse_debug,
        "repair_parse_debug": repair_parse_debug,
        "fill_parse_debug": fill_parse_debug,
        "strict_parse_debug": strict_parse_debug,
        "raw": data,
    }
    return out
