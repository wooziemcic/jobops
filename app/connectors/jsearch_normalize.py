from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.jobs_utils import guess_role_bucket, stable_job_id


def _safe_join_location(city: Optional[str], state: Optional[str], country: Optional[str]) -> str:
    parts = [p.strip() for p in [city or "", state or "", country or ""] if p and p.strip()]
    return ", ".join(parts)


def normalize_jsearch_payload(payload: Dict[str, Any]) -> List[dict]:
    """
    Normalize JSearch response into our internal job schema:
      {id,title,company,location,description,apply_url,posted_at,source,role_bucket,raw}
    """
    out: List[dict] = []

    data = payload.get("data") or []
    for it in data:
        title = it.get("job_title") or ""
        company = it.get("employer_name") or ""
        location = _safe_join_location(
            it.get("job_city"),
            it.get("job_state"),
            it.get("job_country"),
        )

        apply_url = (
            it.get("job_apply_link")
            or it.get("job_google_link")
            or it.get("job_offer_expiration_datetime_utc")  # not a link, but keep fallback minimal
        )
        if apply_url and not isinstance(apply_url, str):
            apply_url = ""

        desc = it.get("job_description") or ""
        posted_at = it.get("job_posted_at_datetime_utc") or it.get("job_posted_at_timestamp")

        role_bucket = guess_role_bucket(title, desc)

        jid = stable_job_id("jsearch", company, title, location, str(apply_url or ""))

        out.append(
            {
                "id": jid,
                "title": title.strip(),
                "company": company.strip(),
                "location": location.strip(),
                "description": desc,
                "apply_url": (apply_url or "").strip(),
                "posted_at": posted_at,
                "source": "jsearch",
                "role_bucket": role_bucket,
                "raw": it,
            }
        )

    return out
