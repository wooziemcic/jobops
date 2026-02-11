from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests


DEFAULT_HOST = "jsearch.p.rapidapi.com"


def fetch_jsearch_jobs(
    *,
    query: str,
    rapidapi_key: str,
    country: str = "us",
    page: int = 1,
    num_pages: int = 1,
    host: str = DEFAULT_HOST,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Calls JSearch 'Job Search' endpoint on RapidAPI.
    Returns raw JSON dict.
    """
    url = f"https://{host}/search"
    headers = {
        "x-rapidapi-key": rapidapi_key,
        "x-rapidapi-host": host,
    }
    params = {
        "query": query,
        "page": page,
        "num_pages": num_pages,
        "country": country,
    }

    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_jsearch_job_details(
    *,
    job_id: str,
    rapidapi_key: str,
    country: str = "us",
    host: str = DEFAULT_HOST,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Calls JSearch 'Job Details' endpoint on RapidAPI.
    Returns raw JSON dict.
    """
    url = f"https://{host}/job-details"
    headers = {
        "x-rapidapi-key": rapidapi_key,
        "x-rapidapi-host": host,
    }
    params = {"job_id": job_id, "country": country}

    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()
