# app/pages/02_Job_Feed.py
from __future__ import annotations

import sys
import json
import math
import time
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.storage.jobs_store import write_raw, write_latest_normalized, read_latest_normalized
from app.connectors.rss import fetch_rss_feed, normalize_rss_entries
from app.connectors.jsearch import fetch_jsearch_jobs
from app.connectors.jsearch_normalize import normalize_jsearch_payload
from app.core.jobs_utils import (
    dedupe_jobs,
    is_early_career,
    is_us_job,
    extract_us_state_code,
    ROLE_BUCKETS,
)

st.set_page_config(page_title="Job Feed", layout="wide")

# -------------------------
# Minimal CSS for cards
# -------------------------
st.markdown(
    """
<style>
.job-card {
  border: 1px solid rgba(49,51,63,.2);
  border-radius: 14px;
  padding: 14px 14px;
  background: rgba(255,255,255,0.02);
  height: 210px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.job-title { font-weight: 700; font-size: 15px; line-height: 1.15; margin-bottom: 6px; }
.job-meta { font-size: 12px; opacity: 0.85; margin-bottom: 8px; }
.job-tags { font-size: 12px; opacity: 0.9; margin-bottom: 8px; }
.job-actions { display:flex; gap:10px; align-items:center; }
.small-muted { font-size: 12px; opacity: 0.7; }
hr.soft { border: none; border-top: 1px solid rgba(49,51,63,.12); margin: 10px 0 14px 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Job Feed")
st.caption("Clean grid view + US map. JSearch is primary; RSS is optional.")

# -------------------------
# Sidebar controls (no big blocks on main page)
# -------------------------
with st.sidebar:
    st.header("Controls")

    # Basic filters
    bucket_filter = st.multiselect(
        "Role buckets",
        options=ROLE_BUCKETS,
        default=["DA", "DS", "BA", "Quant", "ML", "AI"],
    )
    keyword_filter = st.text_input("Keyword filter", value="")
    early_only = st.checkbox("Early-career only (0–4 yrs)", value=True)
    us_only = st.checkbox("US only", value=True)

    st.divider()
    st.subheader("Refresh jobs")

    # Secrets prefer
    secret_key = st.secrets.get("RAPIDAPI_KEY", "")
    secret_host = st.secrets.get("RAPIDAPI_HOST", "jsearch.p.rapidapi.com")

    rapidapi_key = st.text_input("RAPIDAPI_KEY", value=secret_key, type="password")
    rapidapi_host = st.text_input("RAPIDAPI_HOST", value=secret_host)
    country = st.text_input("Country", value="us")

    colA, colB = st.columns(2)
    with colA:
        start_page = st.selectbox("Start page", [1, 2, 3, 5, 10], index=0)
    with colB:
        num_pages = st.selectbox("Pages", [1, 2, 3, 5], index=1)

    st.caption("Tip: Keep Pages low to avoid RapidAPI timeouts.")

    # Advanced: RSS feeds collapsed
    with st.expander("RSS (optional)"):
        DEFAULT_FEEDS = [
            ["Amazon (Data/Analytics)", "https://www.amazon.jobs/en/search.rss?category[]=data-science&category[]=analytics"],
            ["Goldman Sachs", "https://www.goldmansachs.com/careers/search/rss"],
            ["JPMorgan Chase", "https://careers.jpmorgan.com/us/en/jobs/rss"],
            ["Capital One", "https://www.capitalonecareers.com/search-jobs/rss"],
        ]
        feeds_text = st.text_area(
            "Curated feeds JSON [[name, url], ...]",
            value=json.dumps(DEFAULT_FEEDS, indent=2),
            height=210,
        )

    refresh = st.button("Refresh now", type="primary")

# -------------------------
# Refresh execution (only on click)
# -------------------------
if refresh:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_jobs = []

    # RSS (optional)
    try:
        feeds = json.loads(feeds_text) if "feeds_text" in locals() and feeds_text.strip() else []
    except Exception:
        feeds = []
        st.warning("RSS feeds JSON invalid. Skipping RSS.")

    for name, url in feeds:
        try:
            rss_result = fetch_rss_feed(url)
            if isinstance(rss_result, tuple) and len(rss_result) == 2:
                parsed, fetch_meta = rss_result
                raw_payload = {"fetch_meta": fetch_meta, "parsed": parsed}
            else:
                parsed = rss_result
                raw_payload = parsed

            write_raw(f"rss_{name}", raw_payload, ts)
            all_jobs.extend(normalize_rss_entries(name, parsed))
        except Exception as e:
            st.warning(f"RSS error '{name}': {e}")

    # JSearch (primary)
    if not rapidapi_key:
        st.error("RAPIDAPI_KEY is missing. Add it in Streamlit secrets or sidebar.")
    else:
        bucket_to_query = {
            "DA": "data analyst",
            "BA": "business analyst",
            "DS": "data scientist",
            "ML": "machine learning engineer",
            "AI": "ai engineer",
            "Quant": "quantitative analyst",
            "Other": "analyst",
        }

        early_tokens = "(entry OR junior OR associate OR analyst OR new grad OR early career OR 0-3 OR 1-3 OR 2-4)"
        selected = bucket_filter or ["DA", "BA", "DS", "ML", "AI", "Quant"]
        selected = [b for b in selected if b in bucket_to_query]

        for b in selected:
            q = f"({bucket_to_query[b]}) AND {early_tokens}"
            if keyword_filter.strip():
                q = f"{q} AND ({keyword_filter.strip()})"

            last_err = None
            for attempt in range(1, 4):
                try:
                    payload = fetch_jsearch_jobs(
                        query=q,
                        rapidapi_key=rapidapi_key.strip(),
                        host=rapidapi_host.strip(),
                        country=country.strip().lower(),
                        page=int(start_page),
                        num_pages=int(num_pages),
                        timeout=60,
                    )
                    write_raw(f"jsearch_{b}", payload, ts)
                    jobs = normalize_jsearch_payload(payload)
                    for j in jobs:
                        j.setdefault("role_bucket", b)
                    all_jobs.extend(jobs)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(0.75 * attempt)

            if last_err is not None:
                st.warning(f"JSearch failed for {b}: {last_err}")

    # Dedupe + store clean
    all_jobs = dedupe_jobs(all_jobs)
    if early_only:
        all_jobs = [j for j in all_jobs if is_early_career(j)]
    if us_only:
        all_jobs = [j for j in all_jobs if is_us_job(j)]

    write_latest_normalized(all_jobs)
    st.success(f"Saved jobs_latest.json with {len(all_jobs)} jobs.")

# -------------------------
# Load + filter for display
# -------------------------
jobs = read_latest_normalized()

if early_only:
    jobs = [j for j in jobs if is_early_career(j)]
if us_only:
    jobs = [j for j in jobs if is_us_job(j)]
if bucket_filter:
    jobs = [j for j in jobs if j.get("role_bucket") in bucket_filter]
if keyword_filter.strip():
    qk = keyword_filter.strip().lower()
    def hit(j):
        return (
            qk in (j.get("title") or "").lower()
            or qk in (j.get("company") or "").lower()
            or qk in (j.get("description") or "").lower()
        )
    jobs = [j for j in jobs if hit(j)]

st.markdown('<hr class="soft"/>', unsafe_allow_html=True)
st.subheader(f"Jobs ({len(jobs)})")

# -------------------------
# Cards: 4-column grid + pagination
# -------------------------
PAGE_SIZE = 16  # 4x4
total_pages = max(1, math.ceil(len(jobs) / PAGE_SIZE))

if "job_feed_page" not in st.session_state:
    st.session_state.job_feed_page = 1

colp1, colp2, colp3 = st.columns([1, 2, 1])
with colp1:
    if st.button("Prev", disabled=(st.session_state.job_feed_page <= 1)):
        st.session_state.job_feed_page -= 1
with colp2:
    st.write(f"Page {st.session_state.job_feed_page} / {total_pages}")
with colp3:
    if st.button("Next", disabled=(st.session_state.job_feed_page >= total_pages)):
        st.session_state.job_feed_page += 1

start = (st.session_state.job_feed_page - 1) * PAGE_SIZE
end = start + PAGE_SIZE
page_jobs = jobs[start:end]

cols = st.columns(4, gap="large")

for idx, j in enumerate(page_jobs):
    c = cols[idx % 4]
    title = (j.get("title") or "").strip()
    company = (j.get("company") or "").strip()
    loc = (j.get("location") or "").strip()
    bucket = (j.get("role_bucket") or "").strip()
    source = (j.get("source") or "").strip()
    apply_url = (j.get("apply_url") or "").strip()
    posted = (j.get("posted_at") or j.get("date") or "").strip()

    with c:
        st.markdown(
            f"""
<div class="job-card">
  <div>
    <div class="job-title">{title or "Untitled role"}</div>
    <div class="job-meta">{company or "Unknown company"} · {loc or "Unknown location"}</div>
    <div class="job-tags"><span class="small-muted">Bucket:</span> {bucket or "-"} · <span class="small-muted">Source:</span> {source or "-"}</div>
    <div class="small-muted">{posted}</div>
  </div>
  <div class="job-actions">
    {"<a href='" + apply_url + "' target='_blank'>Apply</a>" if apply_url.startswith("http") else "<span class='small-muted'>No apply link</span>"}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

# -------------------------
# Map only (no fallback)
# -------------------------
st.markdown('<hr class="soft"/>', unsafe_allow_html=True)
st.subheader("US job density map (by state)")

state_counts = defaultdict(int)
for j in jobs:
    code = extract_us_state_code(j.get("location") or "")
    if code:
        state_counts[code] += 1

if not state_counts:
    st.info("No state-coded locations found yet. This usually means location strings aren’t parseable into state codes.")
else:
    try:
        import pandas as pd
        import plotly.express as px
    except Exception:
        st.error("Plotly is required for the map. Install: `pip install plotly pandas` and keep them in requirements.txt.")
        st.stop()

    df = pd.DataFrame([{"state": k, "count": v} for k, v in state_counts.items()])
    fig = px.choropleth(
        df,
        locations="state",
        locationmode="USA-states",
        color="count",
        scope="usa",
        hover_name="state",
        hover_data={"count": True},
    )
    st.plotly_chart(fig, use_container_width=True)
