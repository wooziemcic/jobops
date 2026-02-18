# app/core/openai_rerank_legacy.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _compact_job(job: Dict[str, Any], max_desc_chars: int = 1400) -> Dict[str, Any]:
    desc = (job.get("description") or "").strip()
    if len(desc) > max_desc_chars:
        desc = desc[:max_desc_chars] + "..."
    return {
        "title": job.get("title") or "",
        "company": job.get("company") or "",
        "location": job.get("location") or "",
        "apply_url": job.get("apply_url") or "",
        "source": job.get("source") or "",
        "posted_at": job.get("posted_at") or job.get("date") or "",
        "description": desc,
    }


def openai_rerank_topn_legacy(
    candidate_profile: Dict[str, Any],
    resume_text: str,
    ranked_rows: List[Dict[str, Any]],
    top_n: int = 25,
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Legacy OpenAI SDK (0.28.x) reranker.

    Input ranked_rows: list of dicts like:
      {"score": base_score, "breakdown": {...}, "matched_skills": [...], "job": {...}}

    Output:
      Adds row["llm"] = {idx, llm_score, llm_reasons, must_have_hits, missing_gaps}
      Adds row["final_score"] in [0,1], combined score
      Sorts by final_score desc
    """
    try:
        import openai
    except Exception as e:
        raise RuntimeError("openai package not installed. pip install openai==0.28.0") from e

    if api_key:
        openai.api_key = api_key

    subset = ranked_rows[: max(1, int(top_n))]

    jobs_payload = []
    for i, row in enumerate(subset, start=1):
        job = row.get("job") if isinstance(row, dict) else None
        if not isinstance(job, dict):
            continue
        jobs_payload.append(
            {
                "idx": i,
                "base_score": float(row.get("score") or 0.0),
                "job": _compact_job(job),
            }
        )

    system = (
        "You are an assistant that scores resume-to-job fit for early-career roles. "
        "Return ONLY valid JSON. No markdown, no extra text."
    )

    user_obj = {
        "task": "Rerank jobs by fit using resume/profile + job description.",
        "return_format": {
            "items": [
                {
                    "idx": 1,
                    "llm_score": 0,
                    "llm_reasons": ["..."],
                    "must_have_hits": ["..."],
                    "missing_gaps": ["..."],
                }
            ]
        },
        "rules": [
            "llm_score must be integer 0..100",
            "Be honest; do not invent candidate experience",
            "Penalize senior roles, 5+ years requirements, PhD/postdoc requirements",
            "Reward clear skills/tool matches and role alignment",
            "Keep reasons short and specific (max 5 items)",
        ],
        "candidate_profile": candidate_profile,
        "resume_text": resume_text[:6000],
        "jobs": jobs_payload,
    }

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj)},
        ],
        temperature=0.2,
        max_tokens=1800,
    )

    content = resp["choices"][0]["message"]["content"] if resp and resp.get("choices") else "{}"

    # Strict JSON parse with a small recovery attempt
    content_str = (content or "").strip()
    try:
        data = json.loads(content_str)
    except Exception:
        # attempt to extract JSON object if model added stray text
        start = content_str.find("{")
        end = content_str.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(content_str[start : end + 1])
        else:
            raise

    items = data.get("items", [])
    by_idx = {it.get("idx"): it for it in items if isinstance(it, dict) and isinstance(it.get("idx"), int)}

    out: List[Dict[str, Any]] = []
    for i, row in enumerate(subset, start=1):
        row2 = dict(row)
        it = by_idx.get(i)
        if it:
            row2["llm"] = it
            llm_norm = max(0.0, min(1.0, (float(it.get("llm_score", 0)) / 100.0)))
            base = float(row2.get("score") or 0.0)
            row2["final_score"] = 0.60 * base + 0.40 * llm_norm
        else:
            row2["llm"] = None
            row2["final_score"] = float(row2.get("score") or 0.0)
        out.append(row2)

    for row in ranked_rows[len(subset) :]:
        row2 = dict(row)
        row2["llm"] = None
        row2["final_score"] = float(row2.get("score") or 0.0)
        out.append(row2)

    out.sort(key=lambda r: float(r.get("final_score") or 0.0), reverse=True)
    return out
