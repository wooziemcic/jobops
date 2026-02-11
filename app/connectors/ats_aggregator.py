from __future__ import annotations

from typing import Dict, Any, List, Optional
import requests

from app.core.jobs_utils import utc_now_iso, stable_job_id, guess_role_bucket


def fetch_ats_jobs(
    *,
    base_url: str,
    headers: Dict[str, str],
    params: Dict[str, Any],
    timeout: int = 30,
) -> Dict[str, Any]:
    r = requests.get(base_url, headers=headers, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def normalize_ats_payload(
    source_name: str,
    payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Generic normalizer.
    Assumes payload contains a list under one of these keys: jobs/results/data/items
    You may need to tweak mapping once you decide the exact ATS aggregator API.
    """
    now_iso = utc_now_iso()

    candidates = None
    for k in ("jobs", "results", "data", "items"):
        if isinstance(payload.get(k), list):
            candidates = payload.get(k)
            break
    if candidates is None and isinstance(payload, list):
        candidates = payload
    if candidates is None:
        candidates = []

    out: List[Dict[str, Any]] = []

    for j in candidates:
        title = (j.get("title") or j.get("jobTitle") or "").strip()
        company = (j.get("company") or j.get("employer") or j.get("companyName") or "").strip()
        location = (j.get("location") or j.get("jobLocation") or "").strip() or None

        apply_url = (j.get("apply_url") or j.get("applyUrl") or j.get("url") or j.get("jobUrl") or "").strip()
        job_url = (j.get("job_url") or j.get("jobUrl") or apply_url).strip() or None

        desc = (j.get("description") or j.get("snippet") or j.get("summary") or "").strip() or None
        posted = (j.get("posted_at") or j.get("datePosted") or j.get("postedDate") or None)

        source_job_id = str(j.get("id") or j.get("jobId") or j.get("reqId") or "") or None

        if not title or not apply_url:
            continue

        out.append({
            "job_id": stable_job_id("ats", source_name, company, title, apply_url),
            "source": f"ats:{source_name}",
            "source_job_id": source_job_id,
            "title": title,
            "company": company or source_name,
            "location": location,
            "remote": None,
            "apply_url": apply_url,
            "job_url": job_url,
            "description": desc,
            "posted_at": posted,
            "first_seen_at": now_iso,
            "role_bucket": guess_role_bucket(title, desc or ""),
            "tags": [],
            "raw": {"ats_item": j},
        })

    return out
