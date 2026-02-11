from __future__ import annotations

from typing import Dict, Any, List, Tuple
import requests
import feedparser

from app.core.jobs_utils import utc_now_iso, stable_job_id, guess_role_bucket


_DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) JobOpsRSS/1.0",
    "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
}


def fetch_rss_feed(feed_url: str, timeout: int = 25) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Fetch RSS via requests (headers) -> parse bytes with feedparser.
    Returns:
      (parsed_safe_dict, fetch_meta_dict)
    """
    meta: Dict[str, Any] = {"url": feed_url}

    try:
        r = requests.get(feed_url, headers=_DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
        meta["status_code"] = r.status_code
        meta["final_url"] = r.url
        meta["content_type"] = r.headers.get("Content-Type", "")
        meta["bytes"] = len(r.content)

        # If it looks like HTML, skip parsing and return 0 entries with a clear reason
        ct = (meta["content_type"] or "").lower()
        head = (r.content[:200] or b"").lower()
        if "text/html" in ct or b"<!doctype html" in head or b"<html" in head:
            meta["error"] = "Fetched HTML (not RSS). Likely blocked/redirected."
            return {
                "feed_url": feed_url,
                "bozo": True,
                "bozo_exception": meta["error"],
                "feed": {},
                "entries": [],
            }, meta

        parsed = feedparser.parse(r.content)

        safe: Dict[str, Any] = {
            "feed_url": feed_url,
            "bozo": bool(getattr(parsed, "bozo", False)),
            "bozo_exception": str(getattr(parsed, "bozo_exception", "")) if getattr(parsed, "bozo", False) else "",
            "feed": dict(parsed.get("feed", {}) or {}),
            "entries": [dict(e) for e in (parsed.get("entries", []) or [])],
        }
        return safe, meta

    except Exception as e:
        meta["error"] = str(e)
        return {
            "feed_url": feed_url,
            "bozo": True,
            "bozo_exception": str(e),
            "feed": {},
            "entries": [],
        }, meta


def normalize_rss_entries(feed_name: str, parsed_safe: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    now_iso = utc_now_iso()

    entries = parsed_safe.get("entries", []) or []
    for e in entries:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        summary = (e.get("summary") or e.get("description") or "").strip()

        if not title or not link:
            continue

        company = (e.get("author") or "").strip() or feed_name
        published = e.get("published") or e.get("updated") or None

        out.append(
            {
                "job_id": stable_job_id("rss", feed_name, title, link),
                "source": f"rss:{feed_name}",
                "source_job_id": None,
                "title": title,
                "company": company,
                "location": None,
                "remote": None,
                "apply_url": link,
                "job_url": link,
                "description": summary if summary else None,
                "posted_at": published,
                "first_seen_at": now_iso,
                "role_bucket": guess_role_bucket(title, summary),
                "tags": [],
                "raw": {"rss_entry": e},
            }
        )

    return out
