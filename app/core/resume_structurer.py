from __future__ import annotations

import json
from typing import Any, Dict, Set

from app.core.llm import chat_json_strict
from app.core.profile_schema import validate_profile_strict


SYSTEM_PROMPT = """You are a strict resume-to-JSON extractor.
Rules:
- Output ONLY valid JSON. No markdown, no commentary.
- You may ONLY use evidence IDs provided in the evidence map.
- Every skill must cite at least one evidence ID.
- Every experience bullet and project bullet must cite at least one evidence ID.
- Do NOT invent skills, experience, degrees, dates, companies, titles, metrics.
- If unsure, omit it. Do not guess.
- Keep skill names short (e.g., 'Python', 'SQL', 'Power BI', 'Tableau', 'Apache Airflow').
"""

USER_PROMPT_TEMPLATE = """Build a candidate_profile JSON using ONLY the evidence map below.
You MUST reference evidence IDs (like "E0001") for every skill and bullet.

Return JSON with this structure:
{{
  "summary": {{
    "headline": "...",
    "seniority": "entry|junior|mid|senior",
    "confidence": 0.0-1.0,
    "evidence": ["E...."]
  }},
  "target_roles": ["Data Analyst", "Data Scientist", "Business Analyst", "Quantitative Analyst", "Machine Learning Engineer", "AI Engineer"],
  "skills": {{
    "analytics_visualization": [{{"name":"...", "evidence":["E...."], "confidence":0.0-1.0}}],
    "programming_data": [{{"name":"...", "evidence":["E...."], "confidence":0.0-1.0}}],
    "statistical_analysis": [{{"name":"...", "evidence":["E...."], "confidence":0.0-1.0}}],
    "cloud_etl": [{{"name":"...", "evidence":["E...."], "confidence":0.0-1.0}}],
    "workflow_collaboration": [{{"name":"...", "evidence":["E...."], "confidence":0.0-1.0}}],
    "other": [{{"name":"...", "evidence":["E...."], "confidence":0.0-1.0}}]
  }},
  "experience": [
    {{
      "company": "...",
      "title": "...",
      "start": "YYYY-MM or YYYY",
      "end": "YYYY-MM or Present",
      "location": "optional",
      "bullets": [{{"text":"...", "evidence":["E...."]}}]
    }}
  ],
  "education": [
    {{
      "institution": "...",
      "degree": "...",
      "graduation": "YYYY or YYYY-MM",
      "gpa": "optional",
      "evidence": ["E...."]
    }}
  ],
  "projects": [
    {{
      "name": "...",
      "evidence": ["E...."],
      "bullets": [{{"text":"...", "evidence":["E...."]}}]
    }}
  ],
  "keywords": ["..."]
}}

Evidence map JSON:
{evidence_map_json}
"""


def build_candidate_profile(
    *,
    evidence_map: Dict[str, Any],
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    evidence_ids: Set[str] = {e["id"] for e in evidence_map.get("evidence", [])}

    user_prompt = USER_PROMPT_TEMPLATE.format(
        evidence_map_json=json.dumps(evidence_map, ensure_ascii=False)
    )

    profile_dict = chat_json_strict(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
        temperature=0.0,
        max_tokens=1800,
    )

    # strict validation (schema + evidence IDs)
    profile = validate_profile_strict(profile_dict, evidence_ids)

    # return as plain dict for saving
    return profile.model_dump()
