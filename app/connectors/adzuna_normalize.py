# FILE: app/connectors/adzuna_normalize.py
from __future__ import annotations

from typing import Any, Dict, List

from app.core.jobs_utils import utc_now_iso, stable_job_id, guess_role_bucket


def normalize_adzuna_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize Adzuna payload into our unified job schema (dicts).
    """
    now_iso = utc_now_iso()
    results = payload.get("results", []) or []

    out: List[Dict[str, Any]] = []
    for r in results:
        title = (r.get("title") or "").strip()
        company = ((r.get("company") or {}).get("display_name") or "").strip()
        location = ((r.get("location") or {}).get("display_name") or "").strip() or None

        # Adzuna typically provides:
        # - redirect_url: apply/redirect link
        # - adref: sometimes
        apply_url = (r.get("redirect_url") or r.get("adref") or r.get("url") or "").strip()
        job_url = (r.get("redirect_url") or "").strip() or None

        desc = (r.get("description") or "").strip() or None

        posted_at = (r.get("created") or r.get("created_at") or None)
        source_job_id = str(r.get("id") or "") or None

        if not title or not apply_url:
            continue

        out.append(
            {
                "job_id": stable_job_id("adzuna", company, title, apply_url),
                "source": "ats:adzuna",
                "source_job_id": source_job_id,
                "title": title,
                "company": company or "Unknown",
                "location": location,
                "remote": None,
                "apply_url": apply_url,
                "job_url": job_url,
                "description": desc,
                "posted_at": posted_at,
                "first_seen_at": now_iso,
                "role_bucket": guess_role_bucket(title, desc or ""),
                "tags": [],
                "raw": {"adzuna_result": r},
            }
        )

    return out
