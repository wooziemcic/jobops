from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any
import re

from app.core.config import get_paths


def _safe_slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-]+", "_", s)   # replace anything not alnum/_/- with _
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "source"

def jobs_root() -> Path:
    paths = get_paths()
    root = paths.jobs
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "normalized").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    return root


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path, default: Any):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def write_raw(connector_name: str, payload: Any, timestamp: str) -> Path:
    root = jobs_root()
    safe_name = _safe_slug(connector_name)
    p = root / "raw" / f"{safe_name}_{timestamp}.json"
    write_json(p, payload)
    return p

def write_latest_normalized(jobs: List[Dict[str, Any]]) -> Path:
    root = jobs_root()
    p = root / "normalized" / "jobs_latest.json"
    write_json(p, {"jobs": jobs})
    return p


def read_latest_normalized() -> List[Dict[str, Any]]:
    root = jobs_root()
    p = root / "normalized" / "jobs_latest.json"
    data = read_json(p, default={"jobs": []})
    return data.get("jobs", [])
