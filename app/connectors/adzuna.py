# FILE: app/connectors/adzuna.py
from __future__ import annotations

from typing import Any, Dict, Optional
import requests


ADZUNA_BASE = "https://api.adzuna.com/v1/api/jobs"


def fetch_adzuna_jobs(
    *,
    country: str,
    page: int,
    app_id: str,
    app_key: str,
    what: str,
    where: str = "United States",
    results_per_page: int = 50,
    sort_by: str = "date",
    timeout: int = 30,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Adzuna Jobs API
    Example:
      GET https://api.adzuna.com/v1/api/jobs/us/search/1?app_id=...&app_key=...&what=...&where=...&sort_by=date

    Returns JSON payload (dict).
    """
    url = f"{ADZUNA_BASE}/{country}/search/{page}"

    params: Dict[str, Any] = {
        "app_id": app_id,
        "app_key": app_key,
        "what": what,
        "where": where,
        "results_per_page": results_per_page,
        "sort_by": sort_by,
        "content-type": "application/json",
    }
    if extra_params:
        params.update(extra_params)

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()
