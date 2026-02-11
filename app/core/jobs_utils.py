from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Iterable, List, Dict, Tuple
from urllib.parse import urlparse
import re


ROLE_BUCKETS = ["DA", "DS", "BA", "Quant", "ML", "AI", "Other"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_job_id(*parts: str) -> str:
    blob = "||".join([p.strip().lower() for p in parts if p is not None])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_company(company: str) -> str:
    c = normalize_text(company)
    c = re.sub(r"\b(inc|llc|ltd|corp|corporation|company|co)\b\.?", "", c).strip()
    c = re.sub(r"\s+", " ", c)
    return c


def normalize_title(title: str) -> str:
    t = normalize_text(title)
    # Remove common noise words lightly
    t = re.sub(r"\b(remote|hybrid|onsite|on-site)\b", "", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t


def domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def guess_role_bucket(title: str, description: str = "") -> str:
    t = normalize_text(title)
    d = normalize_text(description)

    blob = f"{t} {d}"

    # Order matters: more specific first
    if re.search(r"\bquant\b|\bquantitative\b|\balpha\b|\bfactor\b|\bderivatives?\b", blob):
        return "Quant"
    if re.search(r"\bbusiness analyst\b|\bproduct analyst\b|\brequirements\b|\bstakeholders?\b", blob):
        return "BA"
    if re.search(r"\bdata scientist\b|\bdata science\b|\bmodeling\b|\bml\b|\bmachine learning\b", blob):
        return "DS"
    if re.search(r"\bmachine learning engineer\b|\bml engineer\b|\bmlops\b|\bpytorch\b|\btensorflow\b", blob):
        return "ML"
    if re.search(r"\bai\b|\bartificial intelligence\b|\bllm\b|\bgpt\b|\brag\b", blob):
        return "AI"
    if re.search(r"\bdata analyst\b|\banalyst\b|\bpower bi\b|\btableau\b|\bdashboards?\b", blob):
        return "DA"
    return "Other"

SENIOR_TITLE_BLOCK = re.compile(
    r"\b(principal|staff|senior|sr\.?|lead|manager|director|head|vp|vice president|architect)\b",
    re.IGNORECASE,
)

PHD_BLOCK = re.compile(r"\b(phd|doctorate)\b", re.IGNORECASE)

def is_early_career(job: dict) -> bool:
    title = (job.get("title") or "")
    desc = (job.get("description") or "")

    blob = f"{title}\n{desc}"

    # Block obvious senior/PhD signals
    if SENIOR_TITLE_BLOCK.search(title):
        return False
    if PHD_BLOCK.search(blob):
        return False

    # Accept if it contains early-career signals
    if re.search(r"\b(entry|junior|associate|new grad|early career|recent graduate)\b", blob, re.IGNORECASE):
        return True

    # Accept if it explicitly mentions <=4 years
    if re.search(r"\b(0|1|2|3|4)\+?\s*(years|yrs)\b", blob, re.IGNORECASE):
        # reject if it mentions 5+ anywhere
        if re.search(r"\b(5|6|7|8|9|10)\+?\s*(years|yrs)\b", blob, re.IGNORECASE):
            return False
        return True

    # Otherwise: keep it but only if title looks non-senior
    return True

def dedupe_jobs(jobs: List[dict]) -> List[dict]:
    """
    Dedup strategy:
    - Prefer apply_url uniqueness
    - Otherwise fall back to (company,title,location,domain)
    - Keep the 'best' record (has posted_at, longer description)
    """
    by_key: Dict[str, dict] = {}

    for j in jobs:
        apply = (j.get("apply_url") or "").strip()
        title = normalize_title(j.get("title", ""))
        comp = normalize_company(j.get("company", ""))
        loc = normalize_text(j.get("location") or "")
        dom = domain(apply)

        if apply:
            key = f"apply::{apply}"
        else:
            key = f"sig::{comp}::{title}::{loc}::{dom}"

        if key not in by_key:
            by_key[key] = j
            continue

        # Choose the better one
        cur = by_key[key]
        cur_score = (1 if cur.get("posted_at") else 0) + (len(cur.get("description") or "") / 1000.0)
        new_score = (1 if j.get("posted_at") else 0) + (len(j.get("description") or "") / 1000.0)

        if new_score > cur_score:
            by_key[key] = j

    return list(by_key.values())


def is_us_job(job: dict) -> bool:
    """
    Strict US-only filter using location + URL heuristics.
    Keeps jobs that are clearly US; rejects everything else.
    """
    loc = (job.get("location") or "").lower()
    apply_url = (job.get("apply_url") or "").lower()

    # Strong positive signals
    if "united states" in loc or ", usa" in loc or " usa" in loc:
        return True

    # US states / DC common patterns
    us_state_tokens = [
        ", al", ", ak", ", az", ", ar", ", ca", ", co", ", ct", ", de", ", fl", ", ga",
        ", hi", ", ia", ", id", ", il", ", in", ", ks", ", ky", ", la", ", ma", ", md",
        ", me", ", mi", ", mn", ", mo", ", ms", ", mt", ", nc", ", nd", ", ne", ", nh",
        ", nj", ", nm", ", nv", ", ny", ", oh", ", ok", ", or", ", pa", ", ri", ", sc",
        ", sd", ", tn", ", tx", ", ut", ", va", ", vt", ", wa", ", wi", ", wv", ", wy",
        " washington, dc", " washington dc", ", dc", " dc"
    ]
    if any(tok in loc for tok in us_state_tokens):
        return True

    # Weak positive: Adzuna discovery links sometimes lack location; keep only if location missing
    if ("adzuna" in apply_url or "/us/" in apply_url) and not loc.strip():
        return True

    # Strong negative signals
    non_us_markers = [
        "india", "philippines", "canada", "mexico", "uk", "united kingdom", "ireland",
        "germany", "france", "singapore", "australia", "bengaluru", "bangalore",
        "muntinlupa", "manila", "calabarzon"
    ]
    if any(x in loc for x in non_us_markers):
        return False

    # If we can't confidently place it in the US, reject
    return False

# app/core/jobs_utils.py (append)

US_STATE_CODES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","IA","ID","IL","IN","KS","KY","LA","MA","MD",
    "ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VA","VT","WA","WI","WV","WY","DC"
}

US_STATE_NAME_TO_CODE = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT",
    "DELAWARE": "DE", "FLORIDA": "FL", "GEORGIA": "GA",
    "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN",
    "IOWA": "IA", "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA",
    "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE",
    "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM", "NEW YORK": "NY", "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT",
    "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
    "DISTRICT OF COLUMBIA": "DC"
}

def extract_us_state_code(location: str) -> str:
    loc = (location or "").strip().upper()
    if not loc:
        return ""

    # First try 2-letter code
    for code in US_STATE_CODES:
        if f", {code}" in loc or f" {code}" in loc:
            return code

    # Then try full state name
    for state_name, code in US_STATE_NAME_TO_CODE.items():
        if state_name in loc:
            return code

    return ""

